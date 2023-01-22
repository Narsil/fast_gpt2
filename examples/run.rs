use fast_gpt2::{run, Gpt2Error};
#[tokio::main]
async fn main() -> Result<(), Gpt2Error> {
    run().await
}
