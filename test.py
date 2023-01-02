from transformers import pipeline
import datetime

start = datetime.datetime.now()
pipe = pipeline(model="gpt2")
print(f"Loaded in {datetime.datetime.now() - start}")
pipe("test", max_length=21)
print(f"Ran in {datetime.datetime.now() - start}")
