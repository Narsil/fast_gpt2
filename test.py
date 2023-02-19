import datetime

start = datetime.datetime.now()
import torch

print(f"Loaded torch {datetime.datetime.now() - start}")
torch.zeros((2, 2)).cuda()
print(f"Loaded torch (cuda) {datetime.datetime.now() - start}")


from transformers import pipeline

print(f"Loaded transformers {datetime.datetime.now() - start}")


pipe = pipeline(task="text-generation", model="gpt2-medium", do_sample=False, device=0, revision="refs/pr/3")
pipe.model.config.max_length = None
print(f"Loaded in {datetime.datetime.now() - start}")
inf_start = datetime.datetime.now()
new_tokens = 10
out = pipe("My name was", max_length=3 + new_tokens)
print(f"Tokens: {(datetime.datetime.now() - inf_start)/new_tokens}/tokens")
print(f"Inference took: {(datetime.datetime.now() - inf_start)}")
print(out)
print(f"Ran in {(datetime.datetime.now() - start)}")
