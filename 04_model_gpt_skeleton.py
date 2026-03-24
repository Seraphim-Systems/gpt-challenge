# ============================================================
# 04_model_gpt_skeleton.py
# TinyGPT scaffold
# Students should complete the forward pass and generation
# functions by mirroring TinyBERT and adapting to
# autoregressive language modeling.
# ============================================================


class TinyGPT(nn.Module):
    """
    Decoder-only transformer with a language-modeling head.

    Main ideas:
    1. Use token embeddings and positional embeddings
    2. Pass the sequence through decoder blocks
    3. Project hidden states to vocabulary logits
    4. If targets are provided, compute next-token loss
    """
    def __init__(self, vocab_size, d_model, context_length, n_layers):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.blocks = nn.Sequential(*[
            DecoderBlock(d_model, context_length) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        """
        Parameters
        ----------
        idx : torch.Tensor
            Input token ids of shape [B, T]

        targets : torch.Tensor or None
            Target token ids of shape [B, T]

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits of shape [B, T, vocab_size]

        loss : torch.Tensor or None
            Cross-entropy loss for next-token prediction
        """
        B, T = idx.shape

        # TODO:
        # 1. Compute token embeddings
        # 2. Build position indices
        # 3. Compute positional embeddings
        # 4. Add token and positional embeddings
        # 5. Pass through decoder blocks
        # 6. Apply final layer norm
        # 7. Project to vocabulary logits

        # tok_emb = ...
        # pos = ...
        # pos_emb = ...
        # x = ...
        # x = ...
        # x = ...
        # logits = ...

        loss = None

        if targets is not None:
            # TODO:
            # Flatten logits and targets so they can be used
            # with cross-entropy for next-token prediction.

            # B, T, V = logits.shape
            # logits_flat = ...
            # targets_flat = ...
            # loss = ...

            pass

        return logits, loss

    def generate_greedy(self, idx, max_new_tokens=100):
        """
        Greedy decoding:
        always choose the most probable next token.

        Parameters
        ----------
        idx : torch.Tensor
            Starting sequence of shape [B, T]

        max_new_tokens : int
            Number of tokens to generate

        Returns
        -------
        idx : torch.Tensor
            Extended sequence
        """
        for _ in range(max_new_tokens):
            # TODO:
            # 1. Keep only the last self.context_length tokens
            # 2. Run the model forward
            # 3. Keep logits from the last time step
            # 4. Choose the most probable next token
            # 5. Append it to idx

            # idx_cond = ...
            # logits, _ = ...
            # logits_last = ...
            # next_token = ...
            # idx = ...

            pass

        return idx

    def generate_temperature(self, idx, max_new_tokens=100, temperature=1.0):
        """
        Sampling with temperature.

        temperature < 1.0 makes output more conservative.
        temperature > 1.0 makes output more random.
        """
        for _ in range(max_new_tokens):
            # TODO:
            # 1. Keep only the last self.context_length tokens
            # 2. Run the model forward
            # 3. Keep logits from the last time step
            # 4. Divide logits by temperature
            # 5. Convert to probabilities with softmax
            # 6. Sample the next token
            # 7. Append it to idx

            # idx_cond = ...
            # logits, _ = ...
            # logits_last = ...
            # probs = ...
            # next_token = ...
            # idx = ...

            pass

        return idx

    def generate_top_k(self, idx, max_new_tokens=100, temperature=1.0, k=5):
        """
        Top-k sampling:
        keep only the k most likely tokens and sample from them.
        """
        for _ in range(max_new_tokens):
            # TODO:
            # Suggested steps:
            # 1. Keep only the last self.context_length tokens
            # 2. Run the model forward
            # 3. Keep logits from the last time step and scale by temperature
            # 4. Extract the top-k logits and indices
            # 5. Build filtered logits filled with -inf
            # 6. Put back the top-k values in their original positions
            # 7. Softmax the filtered logits
            # 8. Sample the next token
            # 9. Append it to idx

            # idx_cond = ...
            # logits, _ = ...
            # logits_last = ...
            # topk_vals, topk_idx = ...
            # filtered_logits = ...
            # filtered_logits.scatter_(...)
            # probs = ...
            # next_token = ...
            # idx = ...

            pass

        return idx