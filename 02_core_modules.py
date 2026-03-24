# ============================================================
# 02_core_modules.py
# Core transformer modules
# Attention is fully implemented.
# Other blocks remain partially scaffolded for the activity.
# ============================================================


class FeedForward(nn.Module):
    """
    Position-wise feedforward layer.
    """
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    """
    General attention head that can be used as:

    1. Encoder self-attention
       context=None, causal=False

    2. Decoder self-attention
       context=None, causal=True

    3. Cross-attention
       context=encoder_out, causal=False
    """
    def __init__(self, d_model, context_length, causal=False):
        super().__init__()

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.causal = causal

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x, context=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor that provides the queries. Shape: [B, T, C]

        context : torch.Tensor or None
            Tensor that provides the keys and values. If None,
            self-attention is used and context = x.
            Shape: [B, S, C]

        Returns
        -------
        out : torch.Tensor
            Output of attention. Shape: [B, T, C]
        """
        if context is None:
            context = x

        B, T, C = x.shape
        _, S, _ = context.shape

        Q = self.query(x)          # [B, T, C]
        K = self.key(context)      # [B, S, C]
        V = self.value(context)    # [B, S, C]

        scores = Q @ K.transpose(-2, -1) / math.sqrt(C)   # [B, T, S]

        if self.causal:
            scores = scores.masked_fill(
                self.tril[:T, :S] == 0,
                float("-inf")
            )

        weights = F.softmax(scores, dim=-1)               # [B, T, S]
        out = weights @ V                                 # [B, T, C]

        return out


class EncoderBlock(nn.Module):
    """
    BERT-like block:
    self-attention without causal masking.
    """
    def __init__(self, d_model, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = AttentionHead(d_model, context_length, causal=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """
    GPT-like block:
    masked self-attention + feedforward.
    Students should complete the forward pass
    by mirroring the EncoderBlock structure.
    """
    def __init__(self, d_model, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = AttentionHead(d_model, context_length, causal=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)

    def forward(self, x):
        # TODO:
        # 1. Apply layer norm before masked self-attention
        # 2. Add the residual connection
        # 3. Apply layer norm before feedforward
        # 4. Add the residual connection

        # x = ...
        # x = ...
        return x


class EncoderDecoderBlock(nn.Module):
    """
    BART-like decoder block:
    1. causal self-attention
    2. cross-attention to encoder output
    3. feedforward

    Students should complete the forward pass.
    """
    def __init__(self, d_model, context_length):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = AttentionHead(d_model, context_length, causal=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = AttentionHead(d_model, context_length, causal=False)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)

    def forward(self, x, encoder_out):
        # TODO:
        # 1. Apply causal self-attention with residual connection
        # 2. Apply cross-attention using encoder_out as context
        # 3. Apply feedforward with residual connection

        # x = ...
        # x = ...
        # x = ...
        return x