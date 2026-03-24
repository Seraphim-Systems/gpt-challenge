# ============================================================
# 01_batching.py
# Batch helpers for language modeling, classification, and seq2seq.
# ============================================================

def get_lm_batch(split="train"):
    """
    Batch for GPT-style language modeling.
    x: context window
    y: same window shifted one position to the left
    """
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        low=0,
        high=len(source) - context_length - 1,
        size=(batch_size,)
    )

    x = torch.stack([source[i:i + context_length] for i in starts])
    y = torch.stack([source[i + 1:i + context_length + 1] for i in starts])

    return x.to(device), y.to(device)


def get_classification_batch(split="train"):
    """
    Toy BERT-style classification task.
    Label = 1 if the sequence contains the character 'a', else 0.
    This task is intentionally simple so that the focus stays on architecture.
    """
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        low=0,
        high=len(source) - context_length,
        size=(batch_size,)
    )

    x = torch.stack([source[i:i + context_length] for i in starts])

    labels = []
    for row in x:
        decoded = decode(row.tolist())
        labels.append(1 if "a" in decoded else 0)

    y = torch.tensor(labels, dtype=torch.long)
    return x.to(device), y.to(device)


def get_seq2seq_batch(split="train"):
    """
    Toy BART-style batch.
    Source: chunk of text
    Target input: same chunk
    Target output: chunk shifted left by one character
    This is not a realistic summarization/translation task; it is just
    enough to illustrate encoder-decoder training.
    """
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        low=0,
        high=len(source) - context_length - 1,
        size=(batch_size,)
    )

    src = torch.stack([source[i:i + context_length] for i in starts])
    tgt_in = torch.stack([source[i:i + context_length] for i in starts])
    tgt_out = torch.stack([source[i + 1:i + context_length + 1] for i in starts])

    return src.to(device), tgt_in.to(device), tgt_out.to(device)
