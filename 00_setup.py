# ============================================================
# 00_setup.py
# Setup utilities, tiny corpus, vocabulary, and train/val split
# This fragment intentionally contains no import statements.
# It is designed to be loaded with exec(..., globals()) after the
# main driver has already imported math, random, torch, nn, F, Optional.
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

text = """
Natural language processing is a field of artificial intelligence.
It studies how computers can work with human language.
Some models learn statistical patterns in text.
Other models use neural networks and attention mechanisms.
A language model tries to predict the next token from context.
Transformers do this by combining embeddings, positions, and attention.
Small models can already learn style, repetition, and local structure.
Large models can generate fluent text, but fluency is not the same as understanding.

Students in this class will train a tiny transformer.
The model will read characters and try to predict the next one.
At first the output will look noisy and random.
After training, some structure will begin to appear.
This activity helps us understand how language models work.
""".strip()

text = (text + "\n") * 20

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    """
    Convert a string into token ids.
    """
    return [stoi[c] for c in s]


def decode(ids):
    """
    Convert token ids back into a string.
    """
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)

split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

print("Corpus length:", len(text))
print("Vocabulary size:", vocab_size)
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Sample text:")
print(text[:300])
print("Vocabulary:", chars)

# ============================================================
# Hyperparameters
# Leave these blank for students.
# ============================================================

batch_size = 
context_length = 
d_model = 
n_layers = 
learning_rate = 
