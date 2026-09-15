# SensiSelect: Sensitivity-Guided Interval Selection for Incremental Microarchitecture Design

This Git repository contains the code and cached evaluation outputs for the ICCD 2026 paper:

> **Stop Replacing Simulators: Sensitivity-Guided Interval Selection for Incremental Microarchitecture Design**

**Authors:** Zhenguo Liu, Chengao Shi, Chen Ding, and Jiang Xu.

The dataset and model checkpoints are hosted on Hugging Face and are **not included in this Git repository**:

- **Data (`data/`):** [BlameX/SensiSelect](https://huggingface.co/datasets/BlameX/SensiSelect)
- **Checkpoints (`checkpoints/`):** [BlameX/SensiSelect_Model](https://huggingface.co/BlameX/SensiSelect_Model)

The directory layout and experiment commands below use `data/` and `checkpoints/`
as the local paths for these externally hosted files.

## Directory Structure

```text
SensiSelect_Release/
├── code/                  # Model, training, evaluation, and figure scripts
│   └── seqpred/           # TaoPredictor baseline (model, data loader, train/eval)
├── checkpoints/           # Hosted on Hugging Face; not included in Git
├── data/                  # Hosted on Hugging Face; not included in Git
├── eval_results/          # Cached JSON outputs for paper experiments
├── figures/               # Generated PDF figures
├── requirements.txt       # Python dependencies
├── LICENSE                # Code/model license (PolyForm Noncommercial 1.0.0)
├── LICENSE.data           # Dataset license (CC BY-NC 4.0)
└── README.md
```

## Environment Setup

All commands below assume you run them from the release root:

```bash
cd SensiSelect_Release
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Software versions used for checkpoint evaluation:

- Python 3.12.9
- PyTorch 2.6.0+cu124, CUDA runtime 12.4
- Package versions used for evaluation are listed in `requirements.txt`.

The paper's efficiency measurements use an NVIDIA RTX 5090 with 32 GB VRAM,
PyTorch 2.8.0, and CUDA 12.8.

Hardware guidance:

- GPU with at least 12 GB VRAM for evaluation
- GPU with at least 24 GB VRAM for training

## Data Hosted on Hugging Face

- `data/training_dataset_ALL.pkl` — unified interval-level dataset (labels + workload/hardware features) used by SensiSelect, the absolute regressor, and the endpoint controls
- `data/gem5_hardware_vectors_0_304.json` — hardware vectors for held-out-HW evaluation
- `data/gem5_hardware_vectors_single_param.json` — single-parameter sweep hardware configs
- `data/single_param_sweep_metadata.json` — sweep-pair metadata used by the single-parameter evaluations
- `data/held_out_single_param_configs.txt` — hardware IDs for the 14 held-out single-parameter configurations
- `data/tao_features/` — instruction-level functional-trace features (one subdirectory per benchmark, each with a `manifest.json` and compact `.bin` records) consumed by the **TaoPredictor** baseline. This is the largest item in the release (~43 GB); it is only needed to train or evaluate TaoPredictor and is not used by any SensiSelect path.

Set `SWEEP_ID_WHITELIST` to the IDs in `data/held_out_single_param_configs.txt`
when running the single-parameter evaluations, ablations, and controls, as shown below.

Default data paths are resolved relative to the release root.

Use each script's `--help` output to check its supported data-path arguments.

## Checkpoints Hosted on Hugging Face

### Main Model and Baselines

- `checkpoints/best_model.pth` — final SensiSelect model used for the paper
- `checkpoints/abs_only_regressor_hw_best_model.pth` — absolute-regression baseline
- `checkpoints/taopredictor_best.pt` — TaoPredictor baseline (instruction-level functional-trace model, TAO-style two-layer self-attention; used for the TaoPredictor curve in all three experiments)

### Mechanism-Level Ablations

- `checkpoints/abl_a0_base_jepa_best_model.pth` — Base model (absolute, delta, and dead-zone directional supervision)
- `checkpoints/abl_a1_grouped_rank_jepa_best_model.pth` — Grouped endpoint sensitivity ranking
- `checkpoints/abl_a2_c3a_jepa_best_cache_rank.pth` — Cache-aware ranking score head
- `checkpoints/abl_a3_h1a_jepa_best_cache_rank.pth` — Locality-proxy cache enhancement
- `checkpoints/abl_a4_noh1a_chain_v2_sq_stall_best_model.pth` — Cross-HW-pair global ranking without the locality-proxy path
- `checkpoints/best_model.pth` — Full SensiSelect model

### Endpoint-Anchoring Controls

- `checkpoints/direct_delta_rank_jepa_gmr_best_model.pth` — Direct-Delta
- `checkpoints/endpoint_anchored_rank_jepa_gmr_best_model.pth` — Endpoint-Anchored

## Reproducing the Main Paper Evaluations

The scripts use fixed random seeds. Data-path arguments vary by script; the
commands below show the intended evaluation inputs.

The first run preprocesses the dataset and saves a cache next to the input file.
The commands below overwrite the bundled evaluation JSON files. Use different
output paths to preserve them, and update `--inject-into` and
`--ours-results-path` to point to the corresponding outputs.

### Sensitivity Retrieval

Evaluated on the held-out single-parameter configurations only (`SWEEP_ID_WHITELIST`):

```bash
SWEEP_ID_WHITELIST="$(cat data/held_out_single_param_configs.txt)" \
python code/eval_film_single_param.py \
  --checkpoint checkpoints/best_model.pth \
  --regressor-checkpoint checkpoints/abs_only_regressor_hw_best_model.pth \
  --data-path data/training_dataset_ALL.pkl \
  --sweep-meta-path data/single_param_sweep_metadata.json \
  --sweep-config-path data/gem5_hardware_vectors_single_param.json \
  --output eval_results/eval_single_param_sweep.json
```

The SimPoint/random curves in the retrieval figure come from
`eval_results/eval_single_param_baselines.json`. Regenerate both baselines with
`code/eval_film_baselines.py`, using the same held-out sweep whitelist as above:

```bash
SWEEP_ID_WHITELIST="$(cat data/held_out_single_param_configs.txt)" \
python code/eval_film_baselines.py \
  --data-path data/training_dataset_ALL.pkl \
  --sweep-meta-path data/single_param_sweep_metadata.json \
  --sweep-config-path data/gem5_hardware_vectors_single_param.json \
  --ours-results-path eval_results/eval_single_param_sweep.json \
  --output eval_results/eval_single_param_baselines.json
```

`--ours-results-path` is used only to print a comparison with SensiSelect;
the output JSON contains the SimPoint and random baseline results.

TaoPredictor additionally requires the instruction-level features in
`data/tao_features/`. The following command merges its results into
`eval_single_param_sweep.json`:

```bash
SWEEP_ID_WHITELIST="$(cat data/held_out_single_param_configs.txt)" \
python code/eval_seqpred_single_param.py \
  --seqpred-checkpoint checkpoints/taopredictor_best.pt \
  --data-path data/training_dataset_ALL.pkl \
  --sweep-meta-path data/single_param_sweep_metadata.json \
  --sweep-config-path data/gem5_hardware_vectors_single_param.json \
  --inject-into eval_results/eval_single_param_sweep.json \
  --output eval_results/eval_seqpred_single_param.json
```

### Simulation Budget

```bash
python code/eval_film_simulation_budget.py \
  --checkpoint checkpoints/best_model.pth \
  --regressor-checkpoint checkpoints/abs_only_regressor_hw_best_model.pth \
  --data-path data/training_dataset_ALL.pkl \
  --hw-json-path data/gem5_hardware_vectors_0_304.json \
  --n_pairs 400 --n_hw_groups 13 --tail-frac 0.05 --top-m 3 \
  --multi-selector max_ranknorm \
  --output eval_results/eval_simulation_budget.json
```

Use `--n_hw_groups 13` for both budget evaluators. Their default is 50, but the
held-out configurations provide only 13 valid structured groups. The scripts
raise an error if fewer groups are available than requested; they do not
automatically cap this value. The remaining sampling flags specify the paper's
evaluation configuration.

Run TaoPredictor and merge its results into the evaluation JSON:

```bash
python code/eval_seqpred_simulation_budget.py \
  --seqpred-checkpoint checkpoints/taopredictor_best.pt \
  --data-path data/training_dataset_ALL.pkl \
  --n_pairs 400 --n_hw_groups 13 --tail-frac 0.05 --top-m 3 \
  --multi-selector max_ranknorm \
  --inject-into eval_results/eval_simulation_budget.json \
  --output eval_results/eval_seqpred_simulation_budget.json
```

### Unseen-Hardware Generalization

```bash
python code/eval_film_ood.py \
  --checkpoint checkpoints/best_model.pth \
  --regressor-checkpoint checkpoints/abs_only_regressor_hw_best_model.pth \
  --data-path data/training_dataset_ALL.pkl \
  --score-source routed \
  --output eval_results/eval_hw_generalization.json
```

Run TaoPredictor and merge its results into the evaluation JSON:

```bash
python code/eval_seqpred_ood.py \
  --seqpred-checkpoint checkpoints/taopredictor_best.pt \
  --data-path data/training_dataset_ALL.pkl \
  --inject-into eval_results/eval_hw_generalization.json \
  --output eval_results/eval_seqpred_ood.json
```

### Mechanism-Level Ablations

Evaluated on the same held-out single-parameter configurations. The cross-HW-pair
ranking control omits the locality-proxy path; the full model includes both.

```bash
SWEEP_ID_WHITELIST="$(cat data/held_out_single_param_configs.txt)" \
python code/eval_film_ablation.py \
  --batch_size 256 \
  --data-path data/training_dataset_ALL.pkl \
  --sweep-meta-path data/single_param_sweep_metadata.json \
  --sweep-config-path data/gem5_hardware_vectors_single_param.json \
  --variant a0 "Base Model: Absolute, Delta, and Dead-Zone Directional Supervision" checkpoints/abl_a0_base_jepa_best_model.pth \
  --variant a1 "+ Grouped Endpoint Sensitivity Ranking" checkpoints/abl_a1_grouped_rank_jepa_best_model.pth \
  --variant a2 "+ Cache-Aware Ranking Score Head" checkpoints/abl_a2_c3a_jepa_best_cache_rank.pth \
  --variant a3 "+ Locality-Proxy Cache Enhancement" checkpoints/abl_a3_h1a_jepa_best_cache_rank.pth \
  --variant a4 "Cross-HW-Pair Global Ranking without Locality Proxy" checkpoints/abl_a4_noh1a_chain_v2_sq_stall_best_model.pth \
  --variant a5 "Full SensiSelect" checkpoints/best_model.pth \
  --output eval_results/eval_ablation_results.json
```

### Endpoint-Anchoring Controls

Each control is scored with the same single-parameter retrieval protocol on the
held-out configurations:

```bash
SWEEP_ID_WHITELIST="$(cat data/held_out_single_param_configs.txt)" \
python code/eval_film_single_param.py \
  --checkpoint checkpoints/direct_delta_rank_jepa_gmr_best_model.pth \
  --regressor-checkpoint '' \
  --data-path data/training_dataset_ALL.pkl \
  --sweep-meta-path data/single_param_sweep_metadata.json \
  --sweep-config-path data/gem5_hardware_vectors_single_param.json \
  --output eval_results/eval_direct_delta.json

SWEEP_ID_WHITELIST="$(cat data/held_out_single_param_configs.txt)" \
python code/eval_film_single_param.py \
  --checkpoint checkpoints/endpoint_anchored_rank_jepa_gmr_best_model.pth \
  --regressor-checkpoint '' \
  --data-path data/training_dataset_ALL.pkl \
  --sweep-meta-path data/single_param_sweep_metadata.json \
  --sweep-config-path data/gem5_hardware_vectors_single_param.json \
  --output eval_results/eval_endpoint_anchored.json
```

## Regenerating Figures

The plotting scripts read JSON files from the current directory. Create the
following links before running them:

```bash
ln -sf eval_results/eval_single_param_sweep.json eval_single_param_sweep.json
ln -sf eval_results/eval_single_param_baselines.json eval_single_param_baselines.json
ln -sf eval_results/eval_simulation_budget.json eval_simulation_budget.json
ln -sf eval_results/eval_hw_generalization.json eval_hw_generalization.json
```

Then run:

```bash
python code/fig_retrieval_comparison.py   # -> fig_retrieval_recall.pdf, fig_retrieval_ndcg.pdf
python code/fig_simulation_budget.py      # -> fig_simulation_budget_pairwise.pdf, fig_simulation_budget_multi_hw.pdf
python code/fig_ood_breakdown.py          # -> fig_ood_breakdown.pdf, fig_ood_breakdown_per_metric.pdf
```

The scripts also generate supplementary NDCG, multi-hardware, and per-metric
generalization plots that are not included in the paper.

The mechanism-level ablations (Table II) and endpoint-anchoring controls (Table III) are reported as
tables, not figures; regenerate their numbers with the evaluation commands above.

The generated PDFs will appear in the current working directory unless you move them afterward.

## Reproducing Training

The released `checkpoints/best_model.pth` comes from a two-stage training process:

1. Stage 1: from-scratch endpoint-anchored training with grouped ranking and cross-pair ranking.
2. Stage 2: a chained per-metric recovery schedule warm-started from the Stage 1 checkpoint.

### Stage 1

This command writes outputs into `checkpoints/`:

```bash
SEED=42 \
EXPERIMENT_TAG=checkpoints/release_stage1 \
TORCH_COMPILE=1 \
TRAIN_MODE=dse \
MODEL_TYPE=film \
HW_ENCODER=gam \
GAM_HIDDEN=16 \
HIDDEN=256 \
BATCH_SIZE=1024 \
TRAIN_LR=2e-4 \
TRAIN_WD=1e-4 \
TRAIN_WARMUP_EPOCHS=2 \
TRAIN_EPOCHS=100 \
TRAIN_PATIENCE=15 \
SPLIT_MODE=hw \
PAIR_SAMPLER=random \
SCORE_MODE=cache_rank \
LOSS_METRICS=all \
DSE_LAMBDA_DELTA=1.0 \
DSE_LAMBDA_DIR=2.0 \
DSE_DIR_TEMP=20.0 \
DSE_DEADZONE_Q=30 \
DSE_DEADZONE_CALIB_PAIRS=32768 \
JEPA_LAMBDA=0.01 \
JEPA_STRATIFIED=1 \
JEPA_RATIO=1:1:1 \
JEPA_PAIRS=256 \
L2A_RANK_LAMBDA=0.3 \
L2A_RANK_K=8 \
L2A_RANK_PAIRS=16 \
R1G_ENABLED=1 \
R1G_GROUP_RANK_LAMBDA=0.3 \
R1G_RANK_K=8 \
R1G_RANK_PAIRS=16 \
R1_TRAIN_HEADS=cpi,rob_stall,lq_stall,sq_stall,branch_mpki,btb_miss_rate \
C3A_LAMBDA=0.3 \
C3A_RANK_K=8 \
C3A_RANK_PAIRS=16 \
H1A_ENABLED=1 \
HOTL_LAMBDA=0.0 \
GMR_ENABLED=1 \
GMR_LAMBDA=0.10 \
GMR_PAIRS=256 \
python code/train_unified_v2.py \
  --data-path data/training_dataset_ALL.pkl \
  --hw-json data/gem5_hardware_vectors_0_304.json
```

This produces:

- `checkpoints/release_stage1_best_model.pth`
- `checkpoints/release_stage1_best_cache_rank.pth`

The chained recovery stage starts from `checkpoints/release_stage1_best_cache_rank.pth`.

### Stage 2

Run the chained recovery schedule from the release root:

```bash
PREV=checkpoints/release_stage1_best_cache_rank.pth
for METRIC in l2_mpki l1_mpki branch_mpki btb_miss_rate l1i_mpki cpi l2_miss_rate rob_stall lq_stall sq_stall; do
  TAG="checkpoints/release_chain_${METRIC}"
  SEED=42 \
  EXPERIMENT_TAG="$TAG" \
  TORCH_COMPILE=1 \
  TRAIN_MODE=dse \
  MODEL_TYPE=film \
  HW_ENCODER=gam \
  GAM_HIDDEN=16 \
  HIDDEN=256 \
  BATCH_SIZE=1024 \
  TRAIN_LR=1e-4 \
  TRAIN_WD=1e-4 \
  TRAIN_EPOCHS=100 \
  TRAIN_PATIENCE=5 \
  SPLIT_MODE=hw \
  PAIR_SAMPLER=random \
  SCORE_MODE=single_metric \
  LOSS_METRICS=all \
  DSE_LAMBDA_DELTA=1.0 \
  DSE_LAMBDA_DIR=2.0 \
  DSE_DIR_TEMP=20.0 \
  DSE_DEADZONE_Q=30 \
  DSE_DEADZONE_CALIB_PAIRS=32768 \
  L2A_RANK_LAMBDA=0.3 \
  L2A_RANK_K=8 \
  L2A_RANK_PAIRS=16 \
  R1_ENABLED=1 \
  R1_TRAIN_HEADS="$METRIC" \
  R1_LAMBDA_RANK=0.0 \
  R1G_ENABLED=1 \
  R1G_GROUP_RANK_LAMBDA=0.8 \
  R1G_RANK_K=8 \
  R1G_RANK_PAIRS=16 \
  C3A_LAMBDA=0.3 \
  C3A_RANK_K=8 \
  C3A_RANK_PAIRS=16 \
  H1A_ENABLED=1 \
  HOTL_LAMBDA=0.0 \
  GMR_ENABLED=0 \
  INIT_STRICT=0 \
  python code/train_unified_v2.py \
    --data-path data/training_dataset_ALL.pkl \
    --hw-json data/gem5_hardware_vectors_0_304.json \
    --init-ckpt "$PREV"
  PREV="${TAG}_best_model.pth"
done
```

After the final `sq_stall` step, `"$PREV"` is the reproduced final checkpoint.

### TaoPredictor Baseline

TaoPredictor is trained independently of the SensiSelect path. It consumes the
instruction-level functional-trace features in `data/tao_features/` (opcode, source/
destination register-use bitmaps, memory access-distance context, and branch history)
with a TAO-style two-layer self-attention backbone, and is optimized as a conventional
absolute-endpoint regressor. Train it with:

```bash
(
  cd code
  RUN_NAME=taopredictor LOSS=absolute \
  TAO_FEATURES_DIR=../data/tao_features \
  TRAINING_PKL=../data/training_dataset_ALL.pkl \
    python -m seqpred.train
)
```

The best checkpoint is written to `code/seqpred/runs/taopredictor/best.pt`; point the
evaluation commands at it via `--seqpred-checkpoint`, or copy it over
`checkpoints/taopredictor_best.pt`. The data directory and label pickle can also be
overridden with the `TAO_FEATURES_DIR` / `TRAINING_PKL` environment variables (they
default to the release `data/` paths).

## Evaluation Results

The release includes checkpoint-evaluation outputs generated on 2026-09-08.
All single-parameter evaluations are on the 14 held-out configurations:

- `eval_results/eval_single_param_sweep.json` — Sensitivity Retrieval (SensiSelect + Abs. Regressor + TaoPredictor)
- `eval_results/eval_single_param_baselines.json` — SimPoint / random baselines
- `eval_results/eval_simulation_budget.json` — Simulation Budget (SensiSelect + Abs. Regressor + TaoPredictor + SimPoint/random)
- `eval_results/eval_hw_generalization.json` — Unseen-Hardware Generalization, routed scoring (SensiSelect + Abs. Regressor + TaoPredictor + SimPoint/random)
- `eval_results/eval_ablation_results.json` — Mechanism-level ablations (Table II)
- `eval_results/eval_direct_delta.json` — Table III Direct-Delta control
- `eval_results/eval_endpoint_anchored.json` — Table III Endpoint-Anchored control

TaoPredictor results are merged into the three experiment JSONs above by the
`eval_seqpred_*.py` commands; running those also writes standalone
`eval_results/eval_seqpred_*.json` copies.

These are included for inspection and figure regeneration, but the commands above can regenerate fresh outputs.

## Expected Results

The numbers below describe the 2026-09-08 evaluation, except Table II, which
retains the paper's display precision. All single-parameter numbers are on the
14 held-out configurations.

### Sensitivity Retrieval

- SensiSelect — Overall Spearman `0.672`, Overall Recall@5% `0.479`, Cache-Mean Recall@5% `0.531`
- TaoPredictor — Overall Spearman `0.346`, Overall Recall@5% `0.252`, Cache-Mean Recall@5% `0.255`

### Simulation Budget

- At `K=5`, SensiSelect reaches `0.884` overall / `0.751` hard-subset pairwise accuracy; TaoPredictor reaches `0.851` / `0.703`

### Unseen-Hardware Generalization (routed scoring)

| Stratum | SensiSelect cache R@5% | TaoPredictor cache R@5% |
| --- | ---: | ---: |
| Close | 0.624 | 0.47 |
| Medium | 0.700 | 0.52 |
| Far | 0.681 | 0.55 |

### Mechanism-Level Ablations (Table II)

Starting from the base model, the next three variants successively add grouped
endpoint sensitivity ranking, a cache-aware ranking score head, and locality-proxy
cache enhancement. The cross-HW-pair global ranking control omits the locality-proxy
path; the full model includes both mechanisms.

| Variant | Cache Mean ρ | Cache R@5% | All Mean ρ | All R@5% |
| --- | ---: | ---: | ---: | ---: |
| Base model | 0.534 | 0.361 | 0.516 | 0.379 |
| + Grouped endpoint sensitivity ranking | 0.614 | 0.438 | 0.583 | 0.420 |
| + Cache-aware ranking score head | 0.737 | 0.525 | 0.656 | 0.463 |
| + Locality-proxy cache enhancement | 0.728 | 0.484 | 0.646 | 0.423 |
| Cross-HW-pair global ranking without locality proxy | 0.734 | 0.510 | 0.658 | 0.451 |
| Full SensiSelect | 0.746 | 0.531 | 0.672 | 0.479 |

### Endpoint-Anchoring Controls (Table III)

| Control | Cache Mean ρ | Cache R@5% | All Mean ρ | All R@5% |
| --- | ---: | ---: | ---: | ---: |
| Direct-Delta | 0.676 | 0.306 | 0.318 | 0.173 |
| Endpoint-Anchored | 0.746 | 0.536 | 0.659 | 0.474 |

## Citation

```bibtex
@inproceedings{sensiselect2026,
  title={Stop Replacing Simulators: Sensitivity-Guided Interval Selection for Incremental Microarchitecture Design},
  author={Liu, Zhenguo and Shi, Chengao and Ding, Chen and Xu, Jiang},
  booktitle={Proceedings of the IEEE International Conference on Computer Design (ICCD)},
  year={2026}
}
```

## License

- **Code and Model Checkpoints:** Licensed under the [PolyForm Noncommercial License 1.0.0](./LICENSE) (`PolyForm-Noncommercial-1.0.0`). Noncommercial, academic, and personal research use is permitted; commercial use is prohibited without a separate commercial license.
- **Dataset:** Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](./LICENSE.data). Commercial use is prohibited without prior written authorization.
- **Commercial Licensing:** For commercial use, enterprise deployment, or redistribution inquiries, please contact the authors to obtain a commercial license.

Required Notice: Copyright (c) 2016-2026 Jiang Xu and SensiSelect Contributors (https://github.com/Artificial-Intelligence-System-Lab/SensiSelect)
