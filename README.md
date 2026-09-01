# Reasoning causal tracing

Small mechanistic-interpretability experiments on whether Qwen3-0.6B's verbalized reasoning tracks computations that influence its answer.

## Notebooks

- `notebooks/`: research process and experimental record.
  - `01_model_and_toy_sanity_check.ipynb`: model and behavioral checks.
  - `02_causal_localization.ipynb`: input-relation localization and controls.
  - `03_cot_causal_alignment.ipynb`: CoT interventions and controlled relation probe.
- `src/`: standalone reproductions of the main controlled experiments.

The strongest controlled result holds the prompt and teacher-forced continuation text fixed, changes only a layer-3 prompt state during prefill, and patches the residual stream at the same probe position. The clean and corrupted-state baselines favor opposite relation tokens; causal separation becomes clear in later layers. This result concerns an internal prompt-state intervention and does not establish that verbalized CoT is necessary.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python main.py
```

The default is the position-matched relation probe. Experiments can also be run directly:

```bash
python main.py --experiment relation-probe
python main.py --experiment localization
python src/position_matched_relation_probe.py
python src/causal_localization.py
```

The first run downloads `Qwen/Qwen3-0.6B` and requires enough memory for model activations.

## Reproducibility

Experiments use fixed seeds and deterministic decoding where generation is involved. GPU kernels and library versions can still introduce small numerical differences; the identity controls provide an implementation check.
