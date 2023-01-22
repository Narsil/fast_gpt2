use axum::{extract::State, response::IntoResponse, routing::post, Json, Router};
use fast_gpt2::{download::download, model::Gpt2, ops::special_argmax, Gpt2Error};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::net::SocketAddr;
use tokenizers::Tokenizer;
use tower_http::trace::{self, TraceLayer};
use tracing::{span, Level};

#[derive(Clone)]
struct AppState {
    model: Gpt2<'static>,
    tokenizer: Tokenizer,
}

fn leak_buffer(buffer: Mmap) -> &'static [u8] {
    let buffer: &'static mut Mmap = Box::leak(Box::new(buffer));
    buffer
}

#[tokio::main]
async fn main() -> Result<(), Gpt2Error> {
    // initialize tracing
    tracing_subscriber::fmt::init();
    let span = span!(Level::TRACE, "loading");
    let _enter = span.enter();

    let start = std::time::Instant::now();
    let filename = "model.safetensors";
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/model.safetensors";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    let file = File::open(filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let buffer: &'static [u8] = leak_buffer(buffer);
    let tensors: SafeTensors<'static> = SafeTensors::deserialize(buffer)?;
    let tensors: &'static SafeTensors<'static> = Box::leak(Box::new(tensors));
    println!("Safetensors {:?}", start.elapsed());

    let filename = "tokenizer.json";
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/tokenizer.json";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    let tokenizer = Tokenizer::from_file(filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    let num_heads = 12;
    let gpt2 = Gpt2::from_tensors(tensors, num_heads);
    let state = AppState {
        model: gpt2,
        tokenizer,
    };
    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        // .route("/", get(root))
        // `POST /users` goes to `create_user`
        .route("/", post(inference))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(trace::DefaultMakeSpan::new().level(Level::INFO))
                .on_response(trace::DefaultOnResponse::new().level(Level::INFO)),
        )
        .with_state(state);

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}

// the input to our `create_user` handler
#[derive(Deserialize)]
struct Inputs {
    inputs: String,
}

// the output to our `create_user` handler
#[derive(Serialize)]
struct Outputs {
    generated_text: String,
}

async fn inference(
    (State(state), Json(payload)): (State<AppState>, Json<Inputs>),
) -> impl IntoResponse {
    let span = span!(Level::TRACE, "inference request");
    let _enter = span.enter();
    let tokenizer = state.tokenizer;
    let encoded = tokenizer.encode(payload.inputs, false).unwrap();
    let mut ids = encoded.get_ids().to_vec();
    let mut past_key_values = state.model.empty_past_key_values();
    let mut current_ids = ids.clone();
    for _i in 0..20 {
        // let start = std::time::Instant::now();
        let span = span!(Level::TRACE, "inference loop");
        let _enter = span.enter();
        let logits = state.model.forward(&current_ids, &mut past_key_values);
        let new_id = special_argmax(&logits);
        ids.push(new_id as u32);
        current_ids = vec![new_id as u32];
        // println!("Loop in {:?}", start.elapsed());
    }
    // println!("Result {:?}", tokenizer.decode(ids, false));
    // println!("Total Inference {:?}", start.elapsed());
    let generated_text = tokenizer.decode(ids, false).unwrap();
    let output = Outputs { generated_text };
    Json(output)
}
