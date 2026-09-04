from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


vocabulary = tokenizer.get_vocab()


print("Model:")
print(MODEL_NAME)

print("\nVocabulary size:")
print(len(vocabulary))


print("\nFirst 30 vocabulary entries:")

for token, token_id in list(vocabulary.items())[:30]:
    print(f"{token:20} → {token_id}")

print("="*60)

tokens_to_check = [
    "hello",
    "world",
    "you",
    "##ing",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]"
]

print("\nSelected tokens:")

for token in tokens_to_check:

    token_id = vocabulary.get(token)

    print(f"{token:10} → {token_id}")