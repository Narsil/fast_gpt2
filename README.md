# fast_gpt2

#THIS IS NOT PRODUCTION READY NOR EVEN CORRECT (right now)

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


## Comparison

Here is a reference with the same code in Python (ofc python is much more feature complete, so I included just the import times for reference)

```
python test.py (use TRANSFORMERS_OFFLINE=1 to remove potential network slowdown)
```

```
torch imported in 0:00:00.756219
transformers imported in 0:00:00.966711
Loaded in 0:00:02.615420
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Ran in 0:00:02.663160
```
