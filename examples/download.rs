use fast_gpt2::download::download;
use fast_gpt2::Gpt2Error;

#[tokio::main]
async fn main() -> Result<(), Gpt2Error> {
    let filename = "model.safetensors";
    if !std::path::Path::new(filename).exists() {
        let max_files = 100;
        let chunk_size = 10_000_000;
        let url = "https://huggingface.co/Narsil/gpt2/resolve/main/model.safetensors";
        println!("Downloading {url:?} into {filename:?}");
        download(url, filename, max_files, chunk_size).await?;
    }
    Ok(())
}
