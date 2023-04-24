from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,GenerationConfig

line=input("Enter promt: ")
model_name="google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokens =tokenizer(line,return_tensors="pt")

config = GenerationConfig(max_new_tokens=800)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
outputs = model.generate(**tokens, generation_config=config)
print(tokenizer.batch_decode(outputs,skip_special_tokens=True))
