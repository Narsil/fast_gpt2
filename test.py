from transformers import pipeline
import datetime

start = datetime.datetime.now()
pipe = pipeline(model="gpt2", use_cache=False)
print(f"Loaded in {datetime.datetime.now() - start}")
inf_start = datetime.datetime.now()
out = pipe("test", max_length=21, use_cache=False)
print(f"Ran in {(datetime.datetime.now() - start)}")
print(f"Tokens: {(datetime.datetime.now() - inf_start) / 20}/tokens")
