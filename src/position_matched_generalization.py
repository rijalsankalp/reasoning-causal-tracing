"""Generalize the position-matched relation probe across name permutations."""

import csv
import itertools
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer


SEED = 0
MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT_INTERVENTION_LAYER = 3
MAX_ATTEMPTS = 30
TARGET_QUALIFIED = 12
MIN_BASELINE_SEPARATION = 2.0
IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 5e-5
NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank",
    "Grace", "Henry", "Irene", "Jack", "Kate", "Liam",
]


def chat_text(tokenizer, user_prompt, enable_thinking):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def prompts(a_name, b_name, c_name):
    clean = f"""{a_name} is older than {b_name}.
{b_name} is older than {c_name}.

Who is the oldest?
Explain your reasoning and give the final answer."""
    corrupt = clean.replace(
        f"{a_name} is older than {b_name}",
        f"{a_name} is younger than {b_name}",
        1,
    )
    return clean, corrupt


def direct_answer_text(tokenizer, a_name, b_name, c_name, corrupt=False):
    first_relation = "younger" if corrupt else "older"
    direct_prompt = f"""Answer the question with exactly one name.

{a_name} is {first_relation} than {b_name}.
{b_name} is older than {c_name}.

Who is the oldest?"""
    return chat_text(tokenizer, direct_prompt, enable_thinking=False) + "Answer:"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    older_token = model.to_single_token(" older")
    younger_token = model.to_single_token(" younger")
    assert model.to_string(older_token) == " older"
    assert model.to_string(younger_token) == " younger"

    candidates = list(itertools.permutations(NAMES, 3))
    random.Random(SEED).shuffle(candidates)
    hook_names = [
        f"blocks.{layer}.hook_resid_post"
        for layer in range(model.cfg.n_layers)
    ]
    prompt_hook_name = (
        f"blocks.{PROMPT_INTERVENTION_LAYER}.hook_resid_post"
    )

    qualified = []
    rejected = []
    attempted = 0

    for a_name, b_name, c_name in candidates[:MAX_ATTEMPTS]:
        if len(qualified) >= TARGET_QUALIFIED:
            break
        attempted += 1
        label = f"{a_name}>{b_name}>{c_name}"
        reason = None

        try:
            a_token = model.to_single_token(f" {a_name}")
            b_token = model.to_single_token(f" {b_name}")
        except AssertionError:
            reason = "answer name is not a single token"

        clean_prompt, corrupt_prompt = prompts(a_name, b_name, c_name)
        clean_text = chat_text(model.tokenizer, clean_prompt, enable_thinking=True)
        corrupt_text = chat_text(model.tokenizer, corrupt_prompt, enable_thinking=True)
        clean_prompt_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_prompt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)

        if reason is None and clean_prompt_tokens.shape != corrupt_prompt_tokens.shape:
            reason = "clean/corrupted prompt token shapes differ"
        if reason is None:
            changed = (
                clean_prompt_tokens[0] != corrupt_prompt_tokens[0]
            ).nonzero(as_tuple=False).flatten().tolist()
            if len(changed) != 1:
                reason = f"expected one changed prompt token, found {changed}"
            else:
                relation_position = changed[0]
                relation_pair = (
                    model.to_string(clean_prompt_tokens[0, relation_position]).strip(),
                    model.to_string(corrupt_prompt_tokens[0, relation_position]).strip(),
                )
                if relation_pair != ("older", "younger"):
                    reason = f"changed prompt tokens are {relation_pair}"

        if reason is None:
            clean_direct = model.to_tokens(
                direct_answer_text(
                    model.tokenizer, a_name, b_name, c_name, corrupt=False
                ),
                prepend_bos=False,
            )
            corrupt_direct = model.to_tokens(
                direct_answer_text(
                    model.tokenizer, a_name, b_name, c_name, corrupt=True
                ),
                prepend_bos=False,
            )
            if clean_direct.shape != corrupt_direct.shape:
                reason = "direct-readout token shapes differ"
            else:
                with torch.inference_mode():
                    clean_answer_logits = model(clean_direct)[0, -1]
                    corrupt_answer_logits = model(corrupt_direct)[0, -1]
                clean_answer_gap = float(
                    clean_answer_logits[a_token] - clean_answer_logits[b_token]
                )
                corrupt_answer_gap = float(
                    corrupt_answer_logits[a_token] - corrupt_answer_logits[b_token]
                )
                if not (clean_answer_gap > 0 and corrupt_answer_gap < 0):
                    reason = (
                        "behavioral contrast failed: "
                        f"clean={clean_answer_gap:.4f}, "
                        f"corrupt={corrupt_answer_gap:.4f}"
                    )

        if reason is not None:
            rejected.append({"example": label, "reason": reason})
            continue

        continuation = (
            f"<think>\n{b_name} is older than {a_name}. "
            f"Reconsider the original premises. {a_name} is"
        )
        continuation_tokens = model.to_tokens(continuation, prepend_bos=False)
        condition_a_tokens = torch.cat(
            [clean_prompt_tokens, continuation_tokens], dim=1
        )
        condition_b_tokens = torch.cat(
            [clean_prompt_tokens.clone(), continuation_tokens.clone()], dim=1
        )
        assert torch.equal(condition_a_tokens, condition_b_tokens)
        assert torch.equal(
            condition_a_tokens[:, clean_prompt_tokens.shape[1]:],
            condition_b_tokens[:, clean_prompt_tokens.shape[1]:],
        )
        probe_position = condition_a_tokens.shape[1] - 1
        assert probe_position == condition_b_tokens.shape[1] - 1
        assert model.to_string(
            condition_a_tokens[0, probe_position]
        ).strip() == "is"

        with torch.inference_mode():
            _, corrupt_prompt_cache = model.run_with_cache(
                corrupt_prompt_tokens,
                names_filter=[prompt_hook_name],
            )
        corrupt_relation_state = corrupt_prompt_cache[prompt_hook_name][
            0, relation_position
        ].detach().clone()

        def corrupt_prompt_state(activation, hook):
            patched = activation.clone()
            before = patched.clone()
            patched[0, relation_position] = corrupt_relation_state
            changed_positions = (
                (patched != before).any(dim=-1).nonzero(as_tuple=False).tolist()
            )
            assert changed_positions == [[0, relation_position]]
            return patched

        with torch.inference_mode():
            logits_a, cache_a = model.run_with_cache(
                condition_a_tokens, names_filter=hook_names
            )
            with model.hooks(
                fwd_hooks=[(prompt_hook_name, corrupt_prompt_state)]
            ):
                logits_b, cache_b = model.run_with_cache(
                    condition_b_tokens, names_filter=hook_names
                )

        def relation_gap(logits):
            return float(
                logits[0, probe_position, older_token]
                - logits[0, probe_position, younger_token]
            )

        baseline_a = relation_gap(logits_a)
        baseline_b = relation_gap(logits_b)
        separation = baseline_a - baseline_b
        if not (
            baseline_a > 0
            and baseline_b < 0
            and separation >= MIN_BASELINE_SEPARATION
        ):
            rejected.append({
                "example": label,
                "reason": (
                    "probe baseline failed: "
                    f"A={baseline_a:.4f}, B={baseline_b:.4f}, "
                    f"separation={separation:.4f}"
                ),
            })
            continue

        def patch_probe(base_is_b, source_cache, layer):
            hook_name = hook_names[layer]
            source = source_cache[hook_name][0, probe_position].detach().clone()

            def replace_probe(activation, hook):
                patched = activation.clone()
                patched[0, probe_position] = source
                return patched

            hooks = [(hook_name, replace_probe)]
            if base_is_b:
                hooks.insert(0, (prompt_hook_name, corrupt_prompt_state))
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                return relation_gap(model(condition_a_tokens))

        layer_rows = []
        for layer in range(model.cfg.n_layers):
            patched_ab = patch_probe(True, cache_a, layer)
            patched_ba = patch_probe(False, cache_b, layer)
            identity_a = patch_probe(False, cache_a, layer)
            identity_b = patch_probe(True, cache_b, layer)
            layer_rows.append({
                "layer": layer,
                "D_AB": patched_ab,
                "D_BA": patched_ba,
                "T_AB": (patched_ab - baseline_b) / separation,
                "T_BA": (baseline_a - patched_ba) / separation,
                "AB_source_directed": float(patched_ab > baseline_b),
                "BA_source_directed": float(patched_ba < baseline_a),
                "A_identity_error": identity_a - baseline_a,
                "B_identity_error": identity_b - baseline_b,
            })

        max_identity_error = max(
            max(abs(row["A_identity_error"]), abs(row["B_identity_error"]))
            for row in layer_rows
        )
        identity_tolerance = IDENTITY_ATOL + IDENTITY_RTOL * max(
            abs(baseline_a), abs(baseline_b), 1.0
        )
        if max_identity_error >= identity_tolerance:
            rejected.append({
                "example": label,
                "reason": (
                    f"identity error {max_identity_error:.3e} exceeds "
                    f"{identity_tolerance:.3e}"
                ),
            })
            continue

        qualified.append({
            "example": label,
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "clean_D": baseline_a,
            "corrupt_D": baseline_b,
            "separation": separation,
            "probe_position": probe_position,
            "layers": layer_rows,
        })

    if not qualified:
        raise RuntimeError("No examples passed the preregistered criteria.")

    example_rows = []
    for example in qualified:
        peak_ab = max(example["layers"], key=lambda row: row["T_AB"])
        peak_ba = max(example["layers"], key=lambda row: row["T_BA"])
        example_rows.append({
            "example": example["example"],
            "clean_D": example["clean_D"],
            "corrupt_D": example["corrupt_D"],
            "separation": example["separation"],
            "peak_AB_layer": peak_ab["layer"],
            "peak_AB_transfer": peak_ab["T_AB"],
            "peak_BA_layer": peak_ba["layer"],
            "peak_BA_transfer": peak_ba["T_BA"],
        })

    aggregate_rows = []
    for layer in range(model.cfg.n_layers):
        rows = [example["layers"][layer] for example in qualified]
        transfer_ab = [row["T_AB"] for row in rows]
        transfer_ba = [row["T_BA"] for row in rows]
        aggregate_rows.append({
            "layer": layer,
            "mean_T_AB": float(np.mean(transfer_ab)),
            "std_T_AB": float(np.std(
                transfer_ab, ddof=1 if len(transfer_ab) > 1 else 0
            )),
            "mean_T_BA": float(np.mean(transfer_ba)),
            "std_T_BA": float(np.std(
                transfer_ba, ddof=1 if len(transfer_ba) > 1 else 0
            )),
            "fraction_AB_source_directed": float(np.mean([
                row["AB_source_directed"] for row in rows
            ])),
            "fraction_BA_source_directed": float(np.mean([
                row["BA_source_directed"] for row in rows
            ])),
        })

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == "src" else script_dir
    results_dir = root / "results"
    write_csv(
        results_dir / "position_matched_generalization_examples.csv",
        example_rows,
        list(example_rows[0]),
    )
    write_csv(
        results_dir / "position_matched_generalization_layers.csv",
        aggregate_rows,
        list(aggregate_rows[0]),
    )
    write_csv(
        results_dir / "position_matched_generalization_rejections.csv",
        rejected,
        ["example", "reason"],
    )

    layers = np.array([row["layer"] for row in aggregate_rows])
    mean_ab = np.array([row["mean_T_AB"] for row in aggregate_rows])
    std_ab = np.array([row["std_T_AB"] for row in aggregate_rows])
    mean_ba = np.array([row["mean_T_BA"] for row in aggregate_rows])
    std_ba = np.array([row["std_T_BA"] for row in aggregate_rows])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(layers, mean_ab, marker="o", label="A → B")
    ax.fill_between(layers, mean_ab - std_ab, mean_ab + std_ab, alpha=0.2)
    ax.plot(layers, mean_ba, marker="o", label="B → A")
    ax.fill_between(layers, mean_ba - std_ba, mean_ba + std_ba, alpha=0.2)
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.axhline(1, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(range(model.cfg.n_layers))
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("Normalized source-state transfer")
    ax.set_title(
        f"Position-matched relation-probe generalization (n={len(qualified)})"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figure_path = root / "figures" / "position-matched-generalization.png"
    figure_path.parent.mkdir(exist_ok=True)
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(
        f"attempted={attempted} qualified={len(qualified)} "
        f"rejected={len(rejected)}"
    )
    print(
        "criteria: clean answer gap > 0; corrupted answer gap < 0; "
        "probe A > 0; probe B < 0; probe separation >= "
        f"{MIN_BASELINE_SEPARATION}; aligned tokens; identity tolerance"
    )
    print("\nqualified examples:")
    print(
        "example                 clean_D corrupt_D separation "
        "peak_AB(layer,value) peak_BA(layer,value)"
    )
    for row in example_rows:
        print(
            f"{row['example']:<23} {row['clean_D']:>7.3f} "
            f"{row['corrupt_D']:>9.3f} {row['separation']:>10.3f} "
            f"{row['peak_AB_layer']:>2},{row['peak_AB_transfer']:>6.3f} "
            f"{row['peak_BA_layer']:>2},{row['peak_BA_transfer']:>6.3f}"
        )
    print("\nrejected examples:")
    for row in rejected:
        print(f"{row['example']}: {row['reason']}")
    print("\naggregate by layer:")
    print("layer mean_AB sd_AB mean_BA sd_BA frac_AB frac_BA")
    for row in aggregate_rows:
        print(
            f"{row['layer']:>5} {row['mean_T_AB']:>7.3f} "
            f"{row['std_T_AB']:>5.3f} {row['mean_T_BA']:>7.3f} "
            f"{row['std_T_BA']:>5.3f} "
            f"{row['fraction_AB_source_directed']:>7.3f} "
            f"{row['fraction_BA_source_directed']:>7.3f}"
        )
    print(f"\nfigure: {figure_path.relative_to(root)}")

    return {
        "attempted": attempted,
        "qualified": qualified,
        "rejected": rejected,
        "examples": example_rows,
        "aggregate": aggregate_rows,
        "figure": figure_path,
    }


if __name__ == "__main__":
    run()
