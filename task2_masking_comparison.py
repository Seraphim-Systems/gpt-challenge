import copy

import torch


def compare_masking(model, encode, decode, device, prompt="Natural language", max_new_tokens=150):
    """
    Compare GPT generation with and without causal masking.

    Clones the trained model's weights into an identical copy with
    causal=False on every attention head, then generates from the
    same prompt with both and prints the results side by side.

    Parameters
    ----------
    model          : trained TinyGPT (or compatible) model
    encode         : callable str -> list[int]
    decode         : callable list[int] -> str
    device         : torch device string
    prompt         : seed text for generation
    max_new_tokens : tokens to generate per model
    """
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
