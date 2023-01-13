pub mod download;
mod model;
mod ops;
mod tensor;
use crate::download::download;
use crate::model::Gpt2;
use crate::ops::special_argmax;
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
    let filename = "model.safetensors";
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

    let filename = "tokenizer.json";
    if !std::path::Path::new(filename).exists() {
        let url = "https://huggingface.co/gpt2/resolve/main/tokenizer.json";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    let tokenizer = Tokenizer::from_file(filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    let gpt2 = Gpt2::from_tensors(&tensors);
    let string = "My name is";

    let encoded = tokenizer.encode(string, false).unwrap();
    println!("Loaded & encoded {:?}", start.elapsed());
    let mut ids = encoded.get_ids().to_vec();
    for _i in 0..10 {
        let start = std::time::Instant::now();
        let logits = gpt2.forward(&ids);
        let new_id = special_argmax(&logits);
        ids.push(new_id as u32);
        println!("Loop in {:?}", start.elapsed());
    }
    println!("Result {:?}", tokenizer.decode(ids, false));
    println!("Total Inference {:?}", start.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    pub(crate) fn simplify(data: &[f32]) -> Vec<f32> {
        let precision = 3;
        let m = 10.0 * 10.0f32.powf(precision as f32);
        data.iter().map(|x| (x * m).round() / m).collect()
    }
}
