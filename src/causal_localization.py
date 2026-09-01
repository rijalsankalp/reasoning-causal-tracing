"""Run the Phase A relation-token residual-stream layer scan."""

import random

import numpy as np
import torch
from transformer_lens import HookedTransformer


SEED = 0
MODEL_NAME = "Qwen/Qwen3-0.6B"
CLEAN_PROMPT = """Answer the question with exactly one name.

Alice is older than Bob.
Bob is older than Charlie.

Who is the oldest?"""
CORRUPT_PROMPT = CLEAN_PROMPT.replace("older than Bob", "younger than Bob", 1)


def readout_text(tokenizer, prompt):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prefix + "Answer:"


def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    clean_tokens = model.to_tokens(
        readout_text(model.tokenizer, CLEAN_PROMPT), prepend_bos=False
    )
    corrupt_tokens = model.to_tokens(
        readout_text(model.tokenizer, CORRUPT_PROMPT), prepend_bos=False
    )
    assert clean_tokens.shape == corrupt_tokens.shape
    changed = (clean_tokens[0] != corrupt_tokens[0]).nonzero().flatten().tolist()
    assert len(changed) == 1, changed
    relation_position = changed[0]
    assert model.to_string(clean_tokens[0, relation_position]).strip() == "older"
    assert model.to_string(corrupt_tokens[0, relation_position]).strip() == "younger"

    alice_token = model.to_single_token(" Alice")
    bob_token = model.to_single_token(" Bob")
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)]

    def answer_gap(logits):
        return float(logits[0, -1, alice_token] - logits[0, -1, bob_token])

    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=hook_names
        )
        corrupt_logits, corrupt_cache = model.run_with_cache(
            corrupt_tokens, names_filter=hook_names
        )

    clean_gap = answer_gap(clean_logits)
    corrupt_gap = answer_gap(corrupt_logits)
    denominator = clean_gap - corrupt_gap
    assert abs(denominator) > 1e-6

    def patch(base_tokens, source_cache, layer):
        hook_name = hook_names[layer]
        source = source_cache[hook_name][0, relation_position].detach().clone()

        def replace_relation(activation, hook):
            patched = activation.clone()
            patched[0, relation_position] = source
            return patched

        with torch.inference_mode(), model.hooks(
            fwd_hooks=[(hook_name, replace_relation)]
        ):
            return answer_gap(model(base_tokens))

    rows = []
    for layer in range(model.cfg.n_layers):
        clean_to_corrupt = patch(corrupt_tokens, clean_cache, layer)
        clean_identity = patch(clean_tokens, clean_cache, layer)
        corrupt_identity = patch(corrupt_tokens, corrupt_cache, layer)
        rows.append(
            (
                layer,
                clean_to_corrupt,
                (clean_to_corrupt - corrupt_gap) / denominator,
                clean_identity - clean_gap,
                corrupt_identity - corrupt_gap,
            )
        )

    print(f"model: {MODEL_NAME}")
    print(f"seed: {SEED}")
    print(f"relation position: {relation_position}")
    print(f"clean Alice-Bob gap: {clean_gap:.6f}")
    print(f"corrupt Alice-Bob gap: {corrupt_gap:.6f}")
    print("\nlayer   patched_D  recovery  clean_err  corrupt_err")
    for layer, patched, recovery, clean_error, corrupt_error in rows:
        print(
            f"{layer:>5} {patched:>11.4f} {recovery:>9.4f} "
            f"{clean_error:>10.2e} {corrupt_error:>12.2e}"
        )


if __name__ == "__main__":
    run()
