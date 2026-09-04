from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Hello, how are you?"

tokens = tokenizer.tokenize(text)

print("Model:")
print(MODEL_NAME)

print("\nOriginal text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nNumber of tokens:")
print(len(tokens))