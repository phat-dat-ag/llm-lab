# Tokenization Experiments

This document records my experiments and observations while learning how
tokenization works in Large Language Models (LLMs).

---

# Experiment 01 — Basic Tokenization

## Objective

Understand how raw text is converted into tokens.

## Model

`bert-base-uncased`

## Input

```text
Hello, how are you?
```

## Tokens

```text
hello
,
how
are
you
?
```

## Token Count

```text
6
```

## Observations

1. A token does not necessarily correspond to a complete word.
2. Punctuation can become a separate token.
3. A tokenizer converts raw text into a sequence of tokens.
4. Tokenization is dependent on the tokenizer and its vocabulary.
5. Tokens can be converted into integer token IDs.

## Pipeline

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
```

---

# Experiment 02 — Token IDs

## Objective

Understand how tokens are converted into integer IDs.

## Model

`bert-base-uncased`

## Input

```text
Hello, how are you?
```

## Tokens

```text
['hello', ',', 'how', 'are', 'you', '?']
```

## Token IDs

Example:

```text
hello → 7592
,     → 1010
how   → 2129
are   → 2024
you   → 2017
?     → 1029
```

> The exact token IDs depend on the tokenizer and its vocabulary.
> The numbers should not be memorized.

## Token → ID Mapping

Conceptually:

```text
Token
  ↓
Vocabulary lookup
  ↓
Integer Token ID
```

For example:

```text
"hello"
   ↓
7592
```

## Pipeline

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
```

## Observation

The tokenizer first converts raw text into tokens.

Each token can then be mapped to an integer ID using the tokenizer's
vocabulary.

The model does not directly process the human-readable token strings.
Token IDs are used as the representation before the embedding stage.

---

# Experiment 03 — Vocabulary

## Objective

Understand what a vocabulary is and how it connects tokens with token IDs.

## Model

`bert-base-uncased`

## Vocabulary Size

For `bert-base-uncased`, the vocabulary contains approximately:

```text
30,522 tokens
```

The exact vocabulary size can be inspected directly from the tokenizer.

## Concept

A vocabulary is a collection of tokens known by a tokenizer.

Conceptually:

```text
Vocabulary
    │
    ├── token → ID
    ├── token → ID
    ├── token → ID
    ├── token → ID
    └── ...
```

For example:

```text
hello → 7592
you   → 2017
how   → 2129
```

## Token ↔ Token ID

The vocabulary provides a mapping between tokens and integer IDs.

```text
Token
  │
  ▼
Vocabulary
  │
  ▼
Token ID
```

The reverse mapping is also possible:

```text
Token ID
  │
  ▼
Vocabulary
  │
  ▼
Token
```

## Special Tokens

The vocabulary can also contain special tokens.

Examples for BERT include:

```text
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
```

These tokens have special purposes and are different from ordinary
language tokens.

## Observation

1. Every token represented by the tokenizer has an associated ID.
2. Token IDs are determined by the tokenizer's vocabulary.
3. Different tokenizers can have different vocabularies.
4. Therefore, the same token can have different IDs in different
   tokenizers.
5. Vocabulary size is an important property of a tokenizer/model.

---

# Experiment 04 — Comparing Tokenizers

## Objective

Understand that different tokenizers can tokenize the same text
differently.

## Models

```text
bert-base-uncased
bert-base-multilingual-cased
```

## Test Inputs

### English

```text
Hello, how are you?
```

### Vietnamese

```text
Xin chào, tôi đang học LLM.
```

### Japanese

```text
こんにちは。
```

## Experiment

For each tokenizer, observe:

1. The generated tokens.
2. The number of tokens.
3. How punctuation is represented.
4. How words are split.
5. How Vietnamese is represented.
6. How Japanese is represented.
7. The vocabulary size.

## Concept

The same text can produce different token sequences.

```text
Same Text
    │
    ├── Tokenizer A
    │       ↓
    │    Tokens A
    │
    └── Tokenizer B
            ↓
         Tokens B
```

For example:

```text
Text
 ↓
Tokenizer A
 ↓
8 tokens
```

while:

```text
Text
 ↓
Tokenizer B
 ↓
12 tokens
```

## Observation

Tokenization is not universal.

It depends on:

- The tokenizer algorithm.
- The tokenizer vocabulary.
- The model for which the tokenizer was created.
- The languages and data represented during tokenizer training.

A tokenizer that was designed primarily around one language or
language distribution may represent another language less efficiently.

Therefore:

```text
Same text
    ≠
Same tokens
```

when using different tokenizers.

---

# Experiment 05 — Tokenization Experiments

## Objective

Use different types of text to observe how tokenization behaves.

This file is intended to be a sandbox for experimentation.

---

## Experiment 05.1 — English

Test:

```text
Hello
Hello world
Hello, how are you?
This is a language model.
```

Observe:

- Number of tokens.
- Individual tokens.
- Punctuation.
- Relationship between words and tokens.

---

## Experiment 05.2 — Subwords

Test words such as:

```text
playing
played
player
unbelievable
tokenization
```

Observe whether a word is represented by:

```text
one token
```

or:

```text
multiple tokens
```

Conceptually, a word might be represented as:

```text
playing
   ↓
play + ##ing
```

The exact result depends on the tokenizer.

## Observation

A word does not always correspond to exactly one token.

This is one of the important reasons why:

```text
Number of words
      ≠
Number of tokens
```

---

# Experiment 05.3 — Vietnamese

Test:

```text
Xin chào
Xin chào thế giới
Tôi đang học LLM.
Tôi đang học về Large Language Models.
```

Observe:

- Token count.
- Word splitting.
- Punctuation.
- How Vietnamese characters are represented.
- Whether words are split into multiple subword tokens.

---

# Experiment 05.4 — Japanese

Test:

```text
こんにちは
日本語
日本語を勉強しています。
私は日本語を勉強しています。
私はLLMについて勉強しています。
```

Observe:

- Token count.
- How Japanese characters are grouped.
- Whether words are split.
- How Japanese compares with English and Vietnamese.

---

# Experiment 05.5 — Capitalization

Test:

```text
hello
Hello
HELLO
```

Observe whether capitalization affects tokenization.

This experiment is particularly useful when comparing:

```text
bert-base-uncased
```

with:

```text
bert-base-multilingual-cased
```

---

# Experiment 05.6 — Punctuation

Test:

```text
Hello!
Hello?
Hello.
Hello, world.
Hello: world.
Hello-world.
```

Observe how punctuation is represented as tokens.

---

# Experiment 05.7 — Repeated Text

Test:

```text
hello
hellohello
hello hello
hellohellohello
```

Observe how the tokenizer handles repeated text.

---

# Key Concepts Learned So Far

## 1. Raw Text

The original human-readable input.

Example:

```text
Hello, how are you?
```

---

## 2. Tokenizer

A tokenizer converts raw text into a sequence of tokens.

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
```

---

## 3. Token

A token is a unit of text produced by a tokenizer.

Depending on the tokenizer, a token can represent:

- A complete word.
- Part of a word.
- Punctuation.
- A character or character sequence.
- A special token.
- Other text units defined by the tokenizer.

Therefore:

```text
1 word ≠ necessarily 1 token
```

---

## 4. Subword Token

A word can be divided into multiple smaller tokens.

Conceptually:

```text
playing
   ↓
play + ##ing
```

The exact tokenization depends on the tokenizer.

Subword tokenization allows a tokenizer to represent words that may not
exist as complete vocabulary entries.

---

## 5. Vocabulary

A vocabulary is the collection of tokens known by a tokenizer.

Conceptually:

```text
Vocabulary
    │
    ├── Token A → ID
    ├── Token B → ID
    ├── Token C → ID
    └── ...
```

---

## 6. Token ID

A token ID is an integer representing a token in the vocabulary.

Example:

```text
hello
  ↓
7592
```

The exact ID is tokenizer-dependent.

---

## 7. Special Tokens

Special tokens are tokens reserved for specific purposes by a model.

Examples from BERT include:

```text
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
```

Their exact meaning and usage depend on the model architecture and
tokenizer.

---

# Complete Tokenization Pipeline

At this stage, the basic pipeline can be represented as:

```text
User Text
    │
    ▼
Tokenizer
    │
    ▼
Tokens
    │
    ▼
Vocabulary Lookup
    │
    ▼
Token IDs
```

For example:

```text
"Hello, how are you?"
          │
          ▼
      Tokenizer
          │
          ▼
["hello", ",", "how", "are", "you", "?"]
          │
          ▼
[7592, 1010, 2129, 2024, 2017, 1029]
```

---

# Important Relationship

The most important relationship learned so far is:

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Vocabulary
   ↓
Token IDs
```

Or more precisely:

```text
Raw Text
   │
   ▼
Tokenization
   │
   ▼
Token Sequence
   │
   ▼
Vocabulary Mapping
   │
   ▼
Integer Token IDs
```

---

# Important Observations

## Observation 1 — Token ≠ Word

A token is not necessarily a complete word.

```text
Word
 ↓
One or more tokens
```

---

## Observation 2 — Token Count ≠ Word Count

For example:

```text
Words
 ↓
Tokenization
 ↓
Possibly more tokens
```

Therefore, LLM context limits are normally measured in tokens rather
than words.

---

## Observation 3 — Tokenization Depends on the Model

Different models can use different tokenizers and vocabularies.

Therefore:

```text
Same text
   ↓
Different tokenizer
   ↓
Potentially different tokens
```

---

## Observation 4 — Token IDs Are Not Meaningful Numbers

A token ID such as:

```text
7592
```

does not mean that the token is "more important" than:

```text
2017
```

The number is primarily an index into the vocabulary.

Conceptually:

```text
Token ID
   ↓
Vocabulary / Embedding lookup
   ↓
Representation
```

The semantic representation comes later through embeddings and model
parameters.

---

## Observation 5 — Vocabulary Determines Available Tokens

A tokenizer can only directly represent tokens that exist in its
vocabulary.

If text cannot be represented efficiently using existing vocabulary
entries, the tokenizer may split it into smaller pieces.

---

# Tokenization and Context Window

An LLM's context window is measured in tokens.

For example, if a model supports:

```text
8,192 tokens
```

this means approximately:

```text
8,192 tokens
```

not:

```text
8,192 words
```

The actual number of words that fit depends on the language and the
tokenizer.

Conceptually:

```text
Prompt
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Token Count
   │
   ▼
Context Window
```

This explains why tokenization is directly related to the amount of
text an LLM can process.

---

# Current Mental Model

The current understanding of the LLM input pipeline is:

```text
Human
  │
  │  "Explain TCP/IP"
  ▼
Raw Text
  │
  ▼
Tokenizer
  │
  ▼
Tokens
  │
  ▼
Token IDs
  │
  ▼
Embedding
  │
  ▼
Vectors
  │
  ▼
Transformer
```

At the current stage, the focus is only on:

```text
Raw Text
    ↓
Tokenizer
    ↓
Tokens
    ↓
Token IDs
```

The next stage will investigate what happens after token IDs.

---

# Questions To Answer Before Moving On

Before starting the Embedding Lab, I should be able to explain:

1. What is a token?
2. What is the difference between a token and a word?
3. Why can one word become multiple tokens?
4. What is subword tokenization?
5. What is a tokenizer?
6. What is a vocabulary?
7. What is a token ID?
8. Why does a token ID not represent semantic meaning by itself?
9. Why can different models tokenize the same text differently?
10. Why is the context window measured in tokens?
11. What are special tokens?
12. Why are English, Vietnamese, and Japanese potentially tokenized
    differently?
13. What is the relationship between:
    `text → tokens → vocabulary → token IDs`?

---

# Next Topic — Embeddings

After understanding tokenization, the next question is:

> If token IDs are just integers, how can a Transformer understand
> anything about them?

The next pipeline is:

```text
Token IDs
    │
    ▼
Embedding Matrix
    │
    ▼
Embedding Vectors
    │
    ▼
Transformer
```

Questions for the next lab:

1. Why can't the Transformer simply use token IDs directly?
2. What is an embedding?
3. What is an embedding vector?
4. What does a vector represent?
5. What is an embedding matrix?
6. How does a token ID select a vector from the embedding matrix?
7. Why can semantically related tokens have related vector
   representations?
8. How are embeddings learned during model training?