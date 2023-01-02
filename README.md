# fast_gpt2

## Use
```
cargo run --release # you may need to add --features cblas for better performance
```
Caveat: The first run will actually download the models so will definitely be much slower than this.

Speed to load and run 1 forward pass of gpt2 (not fully checked yet)

```
Safetensors 230.903µs
Tokenizer 47.647504ms
Loaded & encoded 47.850854ms
Total Inference 103.579065ms - [2, 768]
# Subsquent loops take 30ms on the same machine
```

This basically loads the model instantly and runs the first forward pass at 56ms instead of ~30ms for the subsequent passes.

