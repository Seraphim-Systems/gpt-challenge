# ============================================================
# 05_training_utils_and_demos.py
# Full helpers and demos for BERT and BART.
# GPT demo intentionally left empty.
# ============================================================

import torch

from task2_masking_comparison import compare_masking


@torch.no_grad()
def estimate_bert_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        accs = []
        for _ in range(eval_iters):
            x, y = get_classification_batch(split)
            logits, loss = model(x, y)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)
        out[split] = {"loss": sum(losses) / len(losses), "acc": sum(accs) / len(accs)}
    model.train()
    return out


@torch.no_grad()
def estimate_bart_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            src, tgt_in, tgt_out = get_seq2seq_batch(split)
            _, loss = model(src, tgt_in, tgt_out)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def estimate_gpt_loss(model, eval_iters=20):
    """
    Students may use this function once TinyGPT is complete.
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_lm_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ============================================================
# BERT demo (full)
# ============================================================

print("\n" + "=" * 60)
print("BERT-LIKE DEMO")
print("=" * 60)

bert_model = TinyBERT(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers,
    n_classes=2,
).to(device)

bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)

for step in range(101):
    if step % 50 == 0:
        stats = estimate_bert_loss(bert_model, eval_iters=10)
        print(
            f"Step {step:3d} | "
            f"train loss: {stats['train']['loss']:.4f} | "
            f"train acc: {stats['train']['acc']:.4f} | "
            f"val loss: {stats['val']['loss']:.4f} | "
            f"val acc: {stats['val']['acc']:.4f}"
        )

    xb, yb = get_classification_batch("train")
    logits, loss = bert_model(xb, yb)

    bert_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bert_optimizer.step()


# ============================================================
# BART demo (full)
# ============================================================

print("\n" + "=" * 60)
print("BART-LIKE DEMO")
print("=" * 60)

bart_model = TinyBART(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers,
).to(device)

bart_optimizer = torch.optim.Adam(bart_model.parameters(), lr=learning_rate)

for step in range(101):
    if step % 50 == 0:
        losses = estimate_bart_loss(bart_model, eval_iters=10)
        print(
            f"Step {step:3d} | "
            f"train loss: {losses['train']:.4f} | "
            f"val loss: {losses['val']:.4f}"
        )

    src, tgt_in, tgt_out = get_seq2seq_batch("train")
    logits, loss = bart_model(src, tgt_in, tgt_out)

    bart_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bart_optimizer.step()


# ============================================================
# GPT demo (empty on purpose)
# ============================================================

print("\n" + "=" * 60)
print("GPT-LIKE DEMO")
print("=" * 60)

# Completed:
# 1. Instantiate TinyGPT.
# 2. Create an optimizer.
# 3. Train the model with get_lm_batch().
# 4. Evaluate it with estimate_gpt_loss().
# 5. Generate text with temperature, top-k, and beam search.
# 6. Compare outputs qualitatively.



tiny_gpt_model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers
).to(device)

tiny_gpt_optimizer = torch.optim.Adam(tiny_gpt_model.parameters(), lr=learning_rate)

#training
for step in range(101):
    if step % 50 == 0:
        losses = estimate_gpt_loss(tiny_gpt_model, eval_iters=10)
        print(
            f"Step {step:3d} | "
            f"train loss: {losses['train']:.4f} | "
            f"val loss: {losses['val']:.4f}"
        )

    x, y = get_lm_batch("train")
    logits, loss = tiny_gpt_model(x, y)

    tiny_gpt_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    tiny_gpt_optimizer.step()

#evaluation
losses = estimate_gpt_loss(tiny_gpt_model, eval_iters=10)
print(
    f"Final evaluation | "
    f"train loss: {losses['train']:.4f} | "
    f"val loss: {losses['val']:.4f}"
)

#text generation
prompt = "Natural language"
prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

# temperature sampling
temperature_ids = tiny_gpt_model.generate_temperature(
    prompt_ids.clone(), max_new_tokens=80, temperature=1.0
)
temperature_text = decode(temperature_ids[0].tolist())

# top-k sampling
top_k_ids = tiny_gpt_model.generate_top_k(
    prompt_ids.clone(), max_new_tokens=80, temperature=1.0, k=5
)
top_k_text = decode(top_k_ids[0].tolist())

# beam search
beam_ids = tiny_gpt_model.generate_beam_search(
    prompt_ids.clone(), max_new_tokens=80, num_beams=5
)
beam_text = decode(beam_ids[0].tolist())

#comparing outputs
print("\nComparing generated texts:")
print("Temperature sampling:", temperature_text)
print("Top-k sampling:", top_k_text)
print("Beam search:", beam_text)

def _diversity_ratio(text):
    if not text:
        return 0.0
    return len(set(text)) / len(text)

print(
    f"Diversity ratio (unique chars / total chars) -> "
    f"temp: {_diversity_ratio(temperature_text):.3f}, "
    f"top-k: {_diversity_ratio(top_k_text):.3f}, "
    f"beam: {_diversity_ratio(beam_text):.3f}"
)

compare_masking(
    model=tiny_gpt_model,
    encode=encode,
    decode=decode,
    device=device,
    prompt=prompt,
    max_new_tokens=80,
)
