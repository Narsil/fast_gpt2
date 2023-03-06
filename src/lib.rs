pub mod download;
pub mod model;
use crate::download::download;
use crate::model::Gpt2;
#[cfg(feature = "cuda")]
use cudarc::driver::{profiler_start, profiler_stop};
use memmap2::MmapOptions;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use std::fs::File;
use thiserror::Error;
use tokenizers::Tokenizer;

#[cfg(feature = "dfdx")]
use crate::model::dfdx::Dev;

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
    #[cfg(feature = "cuda")]
    #[error("Driver error")]
    ProfilerError(#[from] cudarc::driver::DriverError),
}

pub async fn run() -> Result<(), Gpt2Error> {
    let start = std::time::Instant::now();

    #[cfg(feature = "gpt2")]
    let model_id = "gpt2";
    #[cfg(feature = "gpt2-medium")]
    let model_id = "gpt2-medium";
    #[cfg(feature = "gpt2-large")]
    let model_id = "gpt2-large";

    let filename = format!("model-{model_id}.safetensors");
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(&filename).exists() {
        let revision = "main";

        let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, &filename, max_files, chunk_size).await?;
    }
    let file = File::open(filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;
    println!("Safetensors {:?}", start.elapsed());

    let filename = format!("tokenizer-{model_id}.json");
    if !std::path::Path::new(&filename).exists() {
        let url = format!("https://huggingface.co/{model_id}/resolve/main/tokenizer.json");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, &filename, max_files, chunk_size).await?;
    }
    let tokenizer = Tokenizer::from_file(filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    #[cfg(feature = "dfdx")]
    let dev: Dev = Default::default();

    #[cfg(feature = "gpt2")]
    let num_heads = 12;
    #[cfg(feature = "gpt2-medium")]
    let num_heads = 16;
    #[cfg(feature = "gpt2-large")]
    let num_heads = 20;

    #[cfg(feature = "dfdx")]
    let gpt2 = Gpt2::from_tensors(&tensors, num_heads, &dev);
    #[cfg(not(feature = "dfdx"))]
    let gpt2 = Gpt2::from_tensors(&tensors, num_heads);

    let string = "My name is";

    let encoded = tokenizer.encode(string, false).unwrap();
    println!("Loaded & encoded {:?}", start.elapsed());
    let mut ids = encoded.get_ids().to_vec();

    #[cfg(feature = "dfdx")]
    let mut past_key_values = gpt2.empty_past_key_values(&dev);
    #[cfg(not(feature = "dfdx"))]
    let mut past_key_values = gpt2.empty_past_key_values();

    let mut current_ids = ids.clone();
    #[cfg(feature = "cuda")]
    profiler_start()?;
    for _i in 0..10 {
        // println!("-------------");
        let start = std::time::Instant::now();
        let new_id = gpt2.forward(&current_ids, &mut past_key_values);
        ids.push(new_id as u32);
        current_ids = vec![new_id as u32];
        // #[cfg(feature = "dfdx,cuda")]
        // dev.synchronize().unwrap();
        println!("Loop in {:?}", start.elapsed());
    }
    #[cfg(feature = "cuda")]
    profiler_stop()?;
    println!("Result {:?}", tokenizer.decode(ids, false));
    println!("Total Inference {:?}", start.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use smelt::ops::special_argmax;

    pub(crate) fn simplify(data: &[f32]) -> Vec<f32> {
        let precision = 3;
        let m = 10.0 * 10.0f32.powf(precision as f32);
        data.iter().map(|x| (x * m).round() / m).collect()
    }

    #[test]
    fn simple_logits() {
        let num_heads = 12;
        let filename = "model.safetensors";
        let file = File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();

        let filename = "tokenizer.json";
        let tokenizer = Tokenizer::from_file(filename).unwrap();
        let gpt2 = Gpt2::from_tensors(&tensors, num_heads);
        let string = "My name is";
        let encoded = tokenizer.encode(string, false).unwrap();
        let current_ids = encoded.get_ids().to_vec();
        let mut past_key_values = gpt2.empty_past_key_values();
        let _logits = gpt2.forward(&current_ids, &mut past_key_values);
    }
}
