import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# task2_masking_comparison.py
#
# Exposes compare_masking() -- call it from anywhere by passing
# a trained model, the encode/decode functions, and a prompt.
#
# Standalone usage:
#   python task2_masking_comparison.py
#
# From main.py / another fragment (after TinyGPT is trained):
#   from task2_masking_comparison import compare_masking
#   compare_masking(gpt_model, encode, decode, device)
# ============================================================


def compare_masking(model, encode, decode, device, prompt="Natural language", max_new_tokens=150):
    """
    Compare GPT generation with and without causal masking.

    Clones the trained model's weights into an identical copy with
    causal=False on every attention head, then generates from the
    same prompt with both and prints the results side by side.

    Parameters
    ----------
    model       : trained TinyGPT (or compatible) model
    encode      : callable str -> list[int]
    decode      : callable list[int] -> str
    device      : torch device string
    prompt      : seed text for generation
    max_new_tokens : tokens to generate per model
    """
    # Clone weights; flip causal flag on every attention head
    model_no_mask = copy.deepcopy(model)
    for block in model_no_mask.blocks:
        block.self_attn.causal = False

    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    model.eval()
    model_no_mask.eval()

    with torch.no_grad():
        out_causal  = decode(model.generate_greedy(prompt_ids, max_new_tokens)[0].tolist())
        out_no_mask = decode(model_no_mask.generate_greedy(prompt_ids, max_new_tokens)[0].tolist())

    print("\n" + "=" * 60)
    print("TASK 2 — MASKING COMPARISON")
    print("=" * 60)
    print("Prompt:", repr(prompt))

    print("\n--- WITH causal masking (normal GPT) ---")
    print(out_causal)

    print("\n--- WITHOUT causal masking ---")
    print(out_no_mask)

    print("\n" + "=" * 60)
    print("Look for:")
    print("1. Coherence  — masked output forms recognisable words/phrases;")
    print("               unmasked typically degenerates to repetition/nonsense.")
    print("2. Diversity  — masked output varies across positions; unmasked")
    print("               often collapses to a single repeated token.")
    print("=" * 60)


# ============================================================
# Standalone mode — runs its own training loop so the file
# works independently without needing main.py.
# ============================================================

if __name__ == "__main__":

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

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]

    batch_size    = 16
    context_length = 32
    d_model       = 64
    n_layers      = 2
    learning_rate = 1e-3
    train_steps   = 500

    def get_lm_batch():
        starts = torch.randint(0, len(train_data) - context_length - 1, (batch_size,))
        x = torch.stack([train_data[i : i + context_length] for i in starts])
        y = torch.stack([train_data[i + 1 : i + context_length + 1] for i in starts])
        return x.to(device), y.to(device)

    # --- minimal model definitions for standalone use ---

    class AttentionHead(nn.Module):
        def __init__(self, d_model, context_length, causal=False):
            super().__init__()
            self.query = nn.Linear(d_model, d_model, bias=False)
            self.key   = nn.Linear(d_model, d_model, bias=False)
            self.value = nn.Linear(d_model, d_model, bias=False)
            self.causal = causal
            self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))

        def forward(self, x, context=None):
            if context is None:
                context = x
            B, T, C = x.shape
            _, S, _ = context.shape
            Q = self.query(x)
            K = self.key(context)
            V = self.value(context)
            scores = Q @ K.transpose(-2, -1) / math.sqrt(C)
            if self.causal:
                scores = scores.masked_fill(self.tril[:T, :S] == 0, float("-inf"))
            return F.softmax(scores, dim=-1) @ V

    class FeedForward(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
            )
        def forward(self, x):
            return self.net(x)

    class DecoderBlock(nn.Module):
        def __init__(self, d_model, context_length, causal=True):
            super().__init__()
            self.ln1      = nn.LayerNorm(d_model)
            self.self_attn = AttentionHead(d_model, context_length, causal=causal)
            self.ln2      = nn.LayerNorm(d_model)
            self.ffwd     = FeedForward(d_model)

        def forward(self, x):
            x = x + self.self_attn(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size, d_model, context_length, n_layers):
            super().__init__()
            self.context_length    = context_length
            self.token_embedding   = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(context_length, d_model)
            self.blocks = nn.Sequential(
                *[DecoderBlock(d_model, context_length, causal=True) for _ in range(n_layers)]
            )
            self.ln_f   = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
            x = self.ln_f(self.blocks(x))
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            return logits, loss

        def generate_greedy(self, idx, max_new_tokens=100):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.context_length:]
                logits, _ = self(idx_cond)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_token], dim=1)
            return idx

    # --- train ---
    print("\nTraining TinyGPT...")
    model = TinyGPT(vocab_size, d_model, context_length, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for step in range(train_steps + 1):
        x, y = get_lm_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"  Step {step:4d} | loss: {loss.item():.4f}")

    compare_masking(model, encode, decode, device)
