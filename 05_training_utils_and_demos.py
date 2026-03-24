# ============================================================
# 05_training_utils_and_demos.py
# Full helpers and demos for BERT and BART.
# GPT demo intentionally left empty.
# ============================================================

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
        out[split] = {
            "loss": sum(losses) / len(losses),
            "acc": sum(accs) / len(accs)
        }
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
    n_classes=2
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
    n_layers=n_layers
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

# TODO:
# 1. Instantiate TinyGPT.
# 2. Create an optimizer.
# 3. Train the model with get_lm_batch().
# 4. Evaluate it with estimate_gpt_loss().
# 5. Generate text with:
#       - temperature sampling
#       - top-k sampling
#       - beam search
# 6. Compare the outputs qualitatively.
