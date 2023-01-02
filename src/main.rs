use fast_gpt2::{run, Gpt2Error};

#[tokio::main]
pub async fn main() -> Result<(), Gpt2Error> {
    run().await
}
