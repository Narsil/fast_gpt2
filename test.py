from transformers import pipeline
import datetime

start = datetime.datetime.now()
pipe = pipeline(task="text-generation", model="gpt2", do_sample=False)
pipe.model.config.max_length = None
print(f"Loaded in {datetime.datetime.now() - start}")
inf_start = datetime.datetime.now()
new_tokens = 10
out = pipe("My name is", max_length=3 + new_tokens)
print(f"Ran in {(datetime.datetime.now() - start)}")
print(f"Tokens: {(datetime.datetime.now() - inf_start)/new_tokens}/tokens")
print(out)
