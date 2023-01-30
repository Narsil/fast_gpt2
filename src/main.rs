use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fast_gpt2::{download::download, model::Gpt2, Gpt2Error};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::net::SocketAddr;
use tokenizers::Tokenizer;
use tower_http::trace::{self, TraceLayer};
use tracing::{instrument, Level};

#[derive(Clone)]
struct AppState {
    #[cfg(feature = "dfdx")]
    model: Gpt2,
    #[cfg(not(feature = "dfdx"))]
    model: Gpt2<'static>,
    tokenizer: Tokenizer,
}

fn leak_buffer(buffer: Mmap) -> &'static [u8] {
    let buffer: &'static mut Mmap = Box::leak(Box::new(buffer));
    buffer
}

#[instrument]
async fn get_model(filename: &str) -> Result<Gpt2, Gpt2Error> {
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
    let num_heads = 12;
    Ok(Gpt2::from_tensors(tensors, num_heads))
}

#[instrument]
async fn get_tokenizer(filename: &str) -> Result<Tokenizer, Gpt2Error> {
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/tokenizer.json";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    Ok(Tokenizer::from_file(filename).unwrap())
}

#[tokio::main]
async fn main() -> Result<(), Gpt2Error> {
    // initialize tracing
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "fast_gpt2=debug,tower_http=debug")
    }
    tracing_subscriber::fmt::init();
    let model = get_model("model.safetensors").await?;
    let tokenizer = get_tokenizer("tokenizer.json").await?;

    let state = AppState { model, tokenizer };
    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        // .route("/", get(root))
        // `POST /users` goes to `create_user`
        .route("/", post(inference))
        .route("/", get(health))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(trace::DefaultMakeSpan::new().level(Level::INFO))
                .on_response(trace::DefaultOnResponse::new().level(Level::INFO)),
        )
        .with_state(state);

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()?;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
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

async fn health() -> impl IntoResponse {
    "Ok"
}

async fn inference((State(state), payload): (State<AppState>, String)) -> impl IntoResponse {
    let payload: Inputs = if let Ok(payload) = serde_json::from_str(&payload) {
        payload
    } else {
        Inputs { inputs: payload }
    };
    let tokenizer = state.tokenizer;
    let encoded = tokenizer.encode(payload.inputs, false).unwrap();
    let mut ids = encoded.get_ids().to_vec();
    let mut past_key_values = state.model.empty_past_key_values();
    let mut current_ids = ids.clone();
    for _i in 0..20 {
        let new_id = state.model.forward(&current_ids, &mut past_key_values);
        ids.push(new_id as u32);
        current_ids = vec![new_id as u32];
    }
    let generated_text = tokenizer.decode(ids, false).unwrap();
    let output = Outputs { generated_text };
    Json(vec![output])
}
