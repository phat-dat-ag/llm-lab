from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


texts = [
    "Hello world",
    "Hello, how are you?",
    "I am learning Large Language Models.",
    "Xin chào thế giới.",
    "Tôi đang học về LLM.",
    "こんにちは。",
    "私は日本語を勉強しています。",
    "playing",
    "unbelievable",
    "ChatGPT",
    "tokenization",
]


for text in texts:

    tokens = tokenizer.tokenize(text)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("=" * 80)

    print("TEXT:")
    print(text)

    print("\nTOKENS:")
    print(tokens)

    print("\nTOKEN IDS:")
    print(token_ids)

    print("\nTOKEN COUNT:")
    print(len(tokens))