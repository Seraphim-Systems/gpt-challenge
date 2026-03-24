# ============================================================
# 03_models_bert_bart.py
# Full TinyBERT and TinyBART models.
# Students can mirror these for TinyGPT.
# ============================================================

class TinyBERT(nn.Module):
    """
    Encoder-only transformer with a classification head.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        n_layers: int,
        n_classes: int = 2
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.blocks = nn.Sequential(*[
            EncoderBlock(d_model, context_length) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class TinyBART(nn.Module):
    """
    Encoder-decoder transformer.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        n_layers: int
    ):
        super().__init__()

        self.context_length = context_length

        self.src_token_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_token_embedding = nn.Embedding(vocab_size, d_model)

        self.src_position_embedding = nn.Embedding(context_length, d_model)
        self.tgt_position_embedding = nn.Embedding(context_length, d_model)

        self.encoder_blocks = nn.Sequential(*[
            EncoderBlock(d_model, context_length) for _ in range(n_layers)
        ])

        self.decoder_blocks = nn.ModuleList([
            EncoderDecoderBlock(d_model, context_length) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode(self, src_idx):
        B, T = src_idx.shape

        tok_emb = self.src_token_embedding(src_idx)
        pos = torch.arange(T, device=src_idx.device)
        pos_emb = self.src_position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.encoder_blocks(x)
        return x

    def decode(self, tgt_idx, encoder_out):
        B, T = tgt_idx.shape

        tok_emb = self.tgt_token_embedding(tgt_idx)
        pos = torch.arange(T, device=tgt_idx.device)
        pos_emb = self.tgt_position_embedding(pos)

        x = tok_emb + pos_emb

        for block in self.decoder_blocks:
            x = block(x, encoder_out)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, src_idx, tgt_idx, targets=None):
        encoder_out = self.encode(src_idx)
        logits = self.decode(tgt_idx, encoder_out)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
