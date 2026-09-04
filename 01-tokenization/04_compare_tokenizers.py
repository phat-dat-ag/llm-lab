from transformers import AutoTokenizer


MODELS = [
    "bert-base-uncased",
    "bert-base-multilingual-cased",
]


texts = [
    "Hello, how are you?",
    "Xin chào, tôi đang học LLM.",
    "こんにちは。",
]


for model_name in MODELS:

    print("=" * 80)

    print("MODEL:")
    print(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)

    for text in texts:

        tokens = tokenizer.tokenize(text)

        print("\nText:")
        print(text)

        print("Tokens:")
        print(tokens)

        print("Token count:")
        print(len(tokens))