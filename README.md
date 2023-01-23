# fast_gpt2

Experiment to run from load to finish ML almost 5x faster, works mostly by optimizing load.

[![Fast gpt2 on a real cluster is 3x faster to run](https://img.youtube.com/vi/yqHLIIgOze8/0.jpg)](https://www.youtube.com/watch?v=yqHLIIgOze8)
- Left normal image: https://huggingface.co/Narsil/gpt2: 33s
- Right this repo's image: https://huggingface.co/Narsil/fast_gpt2: 11s

This is an experimental test to remove the need for PyTorch and have a highly specific
runtime that enables to load much faster than using regular PyTorch + transformers using
`safetensors` and direct memory mapping.

## Overview

- Written in Rust
- Almost no dependency (intel-mkl/blas)
- Has a webserver (used to demonstrate differences on real clusters)
- Implements Gpt2 text-generation (greedy mode only) **with past key values** (this is the only way to be on par for performance).
- Docker build (optimized for intel-mkl).
- Docker image **42Mb** (excluding model + tokenizer which get downloaded at runtime, since it's faster than pulling from registry).

## Use
```
cargo run --example run --release --features intel-mkl # for better runtime performance mkl helps
```
Caveat: The first run will actually download the models so will definitely be much slower than this.
Speed to load and run 20 forward passes of gpt2.

```
Safetensors 251.041µs
Tokenizer 43.468349ms
Loaded & encoded 43.681588ms
Loop in 172.272045ms # First loop is slower, no past key values + mmap needs to finish
Loop in 36.165002ms
Loop in 36.269518ms
Loop in 36.311927ms
Loop in 36.329951ms
Loop in 36.477757ms
Loop in 34.368017ms
Loop in 32.67637ms
Loop in 32.67117ms
Loop in 32.909676ms
Result Ok("My name is John. I'm a man of God. I")
Total Inference 530.36737ms
```

This basically loads the model instantly and runs the first forward pass at 56ms instead of ~30ms for the subsequent passes.

## Comparison

Here is a reference with the same code in Python (ofc python is much more feature complete, so I included just the import times for reference)

```
TRANSFORMERS_OFFLINE=1 python test.py (TRANSFORMERS_OFFLINE=1 to remove potential network slowdown)
```

```
Loaded torch 0:00:00.992501
Loaded transformers 0:00:02.095964
Loaded in 0:00:03.444400
/home/nicolas/src/transformers/src/transformers/generation/utils.py:1134: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation)
  warnings.warn(
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Tokens: 0:00:00.081493/tokens
Inference took: 0:00:00.814981
[{'generated_text': "My name is John. I'm a man of God. I"}]
Ran in 0:00:04.259426
```

So almost **5x faster** than the naive PyTorch version. Both use `safetensors` fast loading.
As the logs show, most of the "slow" part is in loading `torch` and `transformers`.
Then the runtime is mostly the same (not here, but it depends on the machine, on most machines I could try runtime
performance was much closer to the point I think they are the same).

Keep in mind this is very naïve PyTorch, there are way to shrink all libs, and make things faster still
The real core important numbers to remember is that this lib is somehow able to load in ~181ms (172+43 for full load + pass - 32ms which is a single pass) compared to ~3.4s from `transformers+pytorch`.


