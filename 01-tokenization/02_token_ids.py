from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Hello, how are you?"

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.convert_tokens_to_ids(tokens)


print("Original text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nToken → ID:")

for token, token_id in zip(tokens, token_ids):
    print(f"{token:10} → {token_id}")