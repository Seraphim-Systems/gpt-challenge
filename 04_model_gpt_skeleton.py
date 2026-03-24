# ============================================================
# 04_model_gpt_skeleton.py
# TinyGPT scaffold
# Students should complete the forward pass and generation
# functions by mirroring TinyBERT and adapting to
# autoregressive language modeling.
# ============================================================

import torch
from torch import nn
import torch.nn.functional as F


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

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None


        if targets is not None:
            # TODO:
            # Flatten logits and targets so they can be used
            # with cross-entropy for next-token prediction.

            # B, T, V = logits.shape
            # logits_flat = ...
            # targets_flat = ...
            # loss = ...
            B, T, V, = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

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

            idx_cond = idx[:, -self.context_length :]
            logits, _ = self.forward(idx_cond)
            logits_last = logits[:, -1, :]
            next_token = torch.argmax(logits_last, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            
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
            
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]           # [B, vocab_size]
            logits_last = logits_last / temperature
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_token], dim=1)

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
            
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :] / temperature   # [B, vocab_size]
            topk_vals, topk_idx = torch.topk(logits_last, k, dim=-1)
            filtered_logits = torch.full_like(logits_last, float("-inf"))
            filtered_logits.scatter_(1, topk_idx, topk_vals)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def generate_beam_search(self, idx, max_new_tokens=100, num_beams=5, length_penalty=0.0):
        """
        Beam search decoding (batch size 1 for simplicity).

        Keeps the top `num_beams` partial hypotheses at each step and
        returns the highest scoring sequence at the end.
        """
        if idx.size(0) != 1:
            raise ValueError("generate_beam_search currently supports batch size 1")

        beams = [(idx, 0.0)]

        for _ in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                idx_cond = seq[:, -self.context_length :]
                logits, _ = self(idx_cond)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, V]

                top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)

                for beam_i in range(num_beams):
                    next_token = top_indices[:, beam_i : beam_i + 1]
                    next_seq = torch.cat([seq, next_token], dim=1)
                    next_score = score + top_log_probs[0, beam_i].item()
                    candidates.append((next_seq, next_score))

            def rank_key(item):
                seq, seq_score = item
                if length_penalty <= 0:
                    return seq_score
                return seq_score / (seq.size(1) ** length_penalty)

            candidates.sort(key=rank_key, reverse=True)
            beams = candidates[:num_beams]

        return beams[0][0]
