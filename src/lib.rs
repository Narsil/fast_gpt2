mod download;
mod model;
mod ops;
mod tensor;
use crate::download::download;
use crate::model::Gpt2;
use memmap2::MmapOptions;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use std::fs::File;
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
    #[error("reqwest header cannot be parsed error")]
    HeaderToStrError(#[from] reqwest::header::ToStrError),
    #[error("Cannot acquire semaphore")]
    AcquireError(#[from] tokio::sync::AcquireError),
    #[error("No content length")]
    NoContentLength,
}

pub async fn run() -> Result<(), Gpt2Error> {
    let start = std::time::Instant::now();
    // curl https://huggingface.co/gpt2/resolve/main/model.safetensors
    let filename = "/tmp/model.safetensors";
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/Narsil/gpt2/resolve/main/model.safetensors";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    let file = File::open(filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;
    println!("Safetensors {:?}", start.elapsed());

    let filename = "/tmp/tokenizer.json";
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/tokenizer.json";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    let tokenizer = Tokenizer::from_file(filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    let gpt2 = Gpt2::from_tensors(&tensors);
    let string = "This is a test";

    let encoded = tokenizer.encode(string, false).unwrap();
    println!("Loaded & encoded {:?}", start.elapsed());
    for _i in 0..5 {
        let start = std::time::Instant::now();
        let _logits = gpt2.forward(encoded.get_ids());
        println!("Inference {:?}", start.elapsed());
    }
    println!("Total Inference {:?}", start.elapsed());
    Ok(())
}
