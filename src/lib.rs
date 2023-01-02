mod model;
mod ops;
mod tensor;
use crate::model::Gpt2;
use futures_util::StreamExt;
use memmap2::MmapOptions;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use std::fs::File;
use std::io::Write;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum Gpt2Error {
    #[error("i/o error")]
    IOError(#[from] std::io::Error),
    #[error("safetensor error")]
    SafeTensorError(#[from] SafeTensorError),
    #[error("slice error")]
    Slice(#[from] std::array::TryFromSliceError),
    #[error("parsing int error")]
    ParseIntError(#[from] core::num::ParseIntError),
    #[error("reqwest int error")]
    RequestError(#[from] reqwest::Error),
}

async fn download(url: &str, filename: &str) -> Result<(), Gpt2Error> {
    let res = reqwest::get(url).await?;
    let mut file = File::create(filename)?;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk)?;
    }
    Ok(())
}

pub async fn run() -> Result<(), Gpt2Error> {
    let start = std::time::Instant::now();
    // curl https://huggingface.co/gpt2/resolve/main/model.safetensors
    let filename = "/tmp/model.safetensors";
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/model.safetensors";
        download(url, filename).await?;
    }

    let file = File::open(&filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;

    println!("Safetensors {:?}", start.elapsed());

    let tokenizer_filename = "/tmp/tokenizer.json";
    if !std::path::Path::new(tokenizer_filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/tokenizer.json";
        download(url, tokenizer_filename).await?;
    }
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    // let dev: Cpu = Default::default();
    // let mut gpt2: TransformerDecoder<768, 12, 1024, 12, Cpu> = dev.build_module();
    // let mut head: Linear<768, 50257> = dev.build_module();
    // println!("Model {:?}", start.elapsed());

    // for (name, view) in tensors.tensors() {
    //     // println!("Name {name:?}");
    //     let tokens: Vec<&str> = name.split(".").into_iter().collect();
    //     let v = view.data();
    //     let data: &[f32] =
    //         unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
    //     match &tokens[..] {
    //         ["h", n, rest @ ..] => {
    //             let n: usize = n.parse()?;
    //             match &rest {
    //                 ["attn", "bias"] => {}
    //                 ["attn", "c_attn", "bias"] => {}
    //                 ["attn", "c_attn", "weight"] => {}
    //                 ["attn", "c_proj", "bias"] => {}
    //                 ["attn", "c_proj", "weight"] => {}
    //                 ["ln_1", "bias"] => {}
    //                 ["ln_1", "weight"] => {}
    //                 ["ln_2", "bias"] => {}
    //                 ["ln_2", "weight"] => {}
    //                 ["mlp", "c_fc", "bias"] => {}
    //                 ["mlp", "c_fc", "weight"] => {}
    //                 ["mlp", "c_proj", "bias"] => {}
    //                 ["mlp", "c_proj", "weight"] => {}
    //                 tokens => {
    //                     panic!("Unhandled weight {tokens:?}")
    //                 }
    //             }
    //         }
    //         ["ln_f", _] => {}
    //         ["wpe", "weight"] => {}
    //         ["wte", "weight"] => {
    //             // head.weight.copy_from(data);
    //         }
    //         tokens => {
    //             panic!("Unhandled weight {tokens:?}")
    //         }
    //     }
    // }
    // println!("Loading {:?}", start.elapsed());
    let gpt2 = Gpt2::from_tensors(&tensors);
    let string = "This is a test";

    let encoded = tokenizer.encode(string, false).unwrap();
    println!("Loaded & encoded {:?}", start.elapsed());
    for i in 0..20 {
        let start = std::time::Instant::now();
        let logits = gpt2.forward(encoded.get_ids());
        // println!("Inference {:?}", start.elapsed());
    }
    println!("Total Inference {:?}", start.elapsed());
    Ok(())
}
