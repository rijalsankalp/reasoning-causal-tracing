"""Run the controlled position-matched relation-probe experiment."""

import random

import numpy as np
import torch
from transformer_lens import HookedTransformer


SEED = 0
MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT_INTERVENTION_LAYER = 3

CLEAN_PROMPT = """Alice is older than Bob.
Bob is older than Charlie.

Who is the oldest?
Explain your reasoning and give the final answer."""
CORRUPT_PROMPT = CLEAN_PROMPT.replace(
    "Alice is older than Bob", "Alice is younger than Bob"
)
CONTINUATION = (
    "<think>\nBob is older than Alice. "
    "Reconsider the original premises. Alice is"
)


def chat_text(tokenizer, user_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    clean_prompt_tokens = model.to_tokens(
        chat_text(model.tokenizer, CLEAN_PROMPT), prepend_bos=False
    )
    corrupt_prompt_tokens = model.to_tokens(
        chat_text(model.tokenizer, CORRUPT_PROMPT), prepend_bos=False
    )
    assert clean_prompt_tokens.shape == corrupt_prompt_tokens.shape

    changed_positions = (
        clean_prompt_tokens[0] != corrupt_prompt_tokens[0]
    ).nonzero(as_tuple=False).flatten().tolist()
    assert len(changed_positions) == 1, changed_positions
    relation_position = changed_positions[0]

    continuation_tokens = model.to_tokens(CONTINUATION, prepend_bos=False)
    full_tokens = torch.cat([clean_prompt_tokens, continuation_tokens], dim=1)
    probe_position = full_tokens.shape[1] - 1

    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)]
    corrupt_hook_name = f"blocks.{PROMPT_INTERVENTION_LAYER}.hook_resid_post"

    with torch.inference_mode():
        _, corrupt_prompt_cache = model.run_with_cache(
            corrupt_prompt_tokens, names_filter=[corrupt_hook_name]
        )

    corrupt_relation_state = corrupt_prompt_cache[corrupt_hook_name][
        0, relation_position
    ].detach().clone()

    def corrupt_prompt_state(activation, hook):
        patched = activation.clone()
        before = patched.clone()
        patched[0, relation_position] = corrupt_relation_state
        changed = (patched != before).any(dim=-1).nonzero(as_tuple=False)
        assert changed.tolist() == [[0, relation_position]]
        return patched

    with torch.inference_mode():
        logits_a, cache_a = model.run_with_cache(full_tokens, names_filter=hook_names)
        with model.hooks(fwd_hooks=[(corrupt_hook_name, corrupt_prompt_state)]):
            logits_b, cache_b = model.run_with_cache(full_tokens, names_filter=hook_names)

    older_token = model.to_single_token(" older")
    younger_token = model.to_single_token(" younger")

    def relation_gap(logits):
        return float(
            logits[0, probe_position, older_token]
            - logits[0, probe_position, younger_token]
        )

    def run_probe_patch(base_is_b, source_cache, layer):
        hook_name = f"blocks.{layer}.hook_resid_post"
        source = source_cache[hook_name][0, probe_position].detach().clone()

        def patch_probe(activation, hook):
            patched = activation.clone()
            patched[0, probe_position] = source
            return patched

        hooks = [(hook_name, patch_probe)]
        if base_is_b:
            hooks.insert(0, (corrupt_hook_name, corrupt_prompt_state))
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            return relation_gap(model(full_tokens))

    baseline_a = relation_gap(logits_a)
    baseline_b = relation_gap(logits_b)
    rows = []
    for layer in range(model.cfg.n_layers):
        rows.append(
            (
                layer,
                run_probe_patch(True, cache_a, layer),
                run_probe_patch(False, cache_b, layer),
                run_probe_patch(False, cache_a, layer),
                run_probe_patch(True, cache_b, layer),
            )
        )

    print(f"model: {MODEL_NAME}")
    print(f"seed: {SEED}")
    print(f"changed prompt position: {relation_position}")
    print(f"probe position: {probe_position}")
    print(f"A baseline older-younger: {baseline_a:.6f}")
    print(f"B baseline older-younger: {baseline_b:.6f}")
    print("\nlayer      A->B      B->A      A->A      B->B")
    for layer, a_to_b, b_to_a, a_to_a, b_to_b in rows:
        print(f"{layer:>5} {a_to_b:>9.4f} {b_to_a:>9.4f} {a_to_a:>9.4f} {b_to_b:>9.4f}")

    max_a_identity_error = max(abs(row[3] - baseline_a) for row in rows)
    max_b_identity_error = max(abs(row[4] - baseline_b) for row in rows)
    print(f"\nmax A->A identity error: {max_a_identity_error:.3e}")
    print(f"max B->B identity error: {max_b_identity_error:.3e}")


if __name__ == "__main__":
    run()
