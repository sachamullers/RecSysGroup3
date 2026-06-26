# LLMInit: Selective Initialization from Large Language Models for Recommendation

This is the PyTorch Implementation for LLMInit.

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n llminit python=3.8
conda activate llminit
```

2. Install PyTorch (adjust CUDA version as needed):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install RecBole and other dependencies:
```bash
pip install recbole
pip install transformers
pip install scikit-learn
```

## Data Processing

The data processing is automatically handled by RecBole framework. When you run the code with our provided configuration:

1. The dataset (e.g., Amazon-Beauty) will be automatically downloaded
2. Data preprocessing will be performed automatically
3. The processed data will be cached for future use

No manual data processing is required.

## Usage Examples

(1) run the **LLMInit-Rand** with the **LightGCN** on Amazon-Beauty
```bash
python run_recbole.py --opt rand -d amazon-beauty -m ContGCN
```

(2) run the **LLMInit-Uni** with the **SGL** on Amazon-Beauty
```bash
python run_recbole.py --opt uni -d amazon-beauty -m ContSGL
```

(3) run the **LLMInit-Var** with the **SGCL** on Amazon-Beauty
```bash
python run_recbole.py --opt var -d amazon-beauty -m ContSGCL
```
## Significance tests on results with multiple seeds

We include a script for running paired one-tailed t-tests with Holm correction across multiple seeds. The required Excel files with the seed results are provided in:

results/multiple_seeds/

Each Excel file contains the results for one dataset, with rows for the baseline models and LLMInit variants, and columns for Recall@10 and NDCG@10 across seeds.

Run the significance tests with:

```bash
python significance_tests.py -d amazon-beauty
```

## Fairness and Diversity Evaluation
We include two scripts for evaluating fairness and diversity from saved recommendation models.

First, export Top-K recommendations from a checkpoint:
```bash
python data_analysis/fairness_eval/export_topk_recommendations.py \
  --model-file saved/checkpoint.pth \
  --output data_analysis/fairness_eval/recommendations/baseline_lightgcn_top20.csv \
  --k 20 \
  --batch-size 128 \
  --external-ids
```

Then compute the fairness metrics:
```bash
python data_analysis/fairness_eval/fairness_metrics_posthoc.py \
  --recommendations data_analysis/fairness_eval/recommendations/baseline_lightgcn_top20.csv \
  --interactions dataset/amazon-beauty/amazon-beauty.inter \
  --items dataset/amazon-beauty/amazon-beauty-without-emb.item \
  --k 10 \
  --tail-buckets 100 \
  --model-name Baseline-LightGCN \
  --output-dir data_analysis/fairness_eval/outputs/baseline_b100_k10
```
The evaluation reports catalog coverage, average brand diversity, average tail-item diversity, brand-loyal versus non-brand-loyal diversity gaps, and recommendation exposure across popularity buckets.

## Connector and Cold-Start Extension

This repository extends the LLMInit reproduction with an item embedding connector for cold-item insertion experiments.

### Reproducibility Quick Start

Run all commands from the repository root:

```bash
cd RecSysGroup3
```

Required prepared datasets are available under:

- `dataset/amazon-beauty/`
- `dataset/amazon-beauty-cold10core/`
- `dataset/amazon-beauty-low5cold/`

### Important Code Changes

1. `run_recbole.py` passes `--opt rand|uni|var` into runtime config as `emb_selection`.
2. `recbole/model/general_recommender/contgcn.py` adds an optional item connector.
3. `recbole/model/general_recommender/embedding_connector.py` defines the connector MLP.
4. Residual connector mode uses:

```text
final item embedding = LLMInit-var + alpha * connector(full LLM item embedding)
```

### Main Connector Flags

Connector behavior is controlled by config values such as:

```yaml
use_connector: True
use_residual_connector: True
residual_connector_alpha: 0.1
connector_align_weight: 0.0
emb_selection: var
```

### Dataset Split Notes

1. Cold-10: random held-out cold items from active items.
2. Low-5: naturally cold items with fewer than 5 interactions.

Current split scripts in active use:

```bash
python utils/make_cold_item_split.py --dataset amazon-beauty --remove_ratio 0.1 --suffix cold10core
python make_pure_cold_low5_split.py
```

`utils/make_cold_item_split.py` is the main Cold-10 split script. It performs seeded random item holdout with `--remove_ratio 0.1` by default.

### Reproduce Cold-10 Main Table

Use `eval_phase2_removed_items.py` with curated checkpoints:

```bash
python eval_phase2_removed_items.py \
  --checkpoint results/checkpoints/cold10_llminit_var_no_connector.pth \
  --cold_dataset amazon-beauty-cold10core \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_core_llminit_var

python eval_phase2_removed_items.py \
  --checkpoint results/checkpoints/cold10_plain_connector_var.pth \
  --cold_dataset amazon-beauty-cold10core \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_core_plain_connector_var

python eval_phase2_removed_items.py \
  --checkpoint results/checkpoints/cold10_residual_connector_var_alpha01.pth \
  --cold_dataset amazon-beauty-cold10core \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_core_residual_connector_var_alpha01
```

### Reproduce Low-5 Appendix Table

```bash
python eval_phase2_removed_items.py \
  --checkpoint results/checkpoints/low5_plain_connector_var_norefilter.pth \
  --cold_dataset amazon-beauty-low5cold \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_low5_plain_connector_var_norefilter

python eval_phase2_removed_items.py \
  --checkpoint results/checkpoints/low5_residual_connector_var_alpha01_norefilter.pth \
  --cold_dataset amazon-beauty-low5cold \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_low5_residual_connector_var_alpha01_norefilter
```

### Reproduce Fairness Connector Rows

1. Export Top-K recommendations with `data_analysis/fairness_eval/export_topk_recommendations.py` using:
   - `results/checkpoints/beauty_plain_connector_var.pth`
   - `results/checkpoints/beauty_residual_connector_var_alpha01.pth`
2. Run `data_analysis/fairness_eval/fairness_metrics_posthoc.py` on each exported recommendation file.

### Key Scripts

1. `utils/make_cold_item_split.py`
2. `make_pure_cold_low5_split.py`
3. `eval_phase2_removed_items.py`
4. `export_phase2_topk_examples.py`
5. `diagnose_connector_scale.py`
6. `inspect_saved_checkpoints.py`
7. `make_final_coldstart_comparison.py`
8. `extract_results.py`

### Checkpoints 

Model checkpoint files (`*.pth`) are not committed to this repository because they are too large for normal GitHub storage. 
The checkpoint files used for the final experiments will be copied to the shared `prjs` project folder. The repository keeps only checkpoint metadata in: `text results/checkpoints/checkpoint_inventory.csv`

### Preserved Reproducibility Checkpoints

The following renamed checkpoints in `results/checkpoints/` are the minimum set used for final claims:

1. `cold10_llminit_var_no_connector.pth`
2. `cold10_plain_connector_var.pth`
3. `cold10_residual_connector_var_alpha01.pth`
4. `low5_plain_connector_var_norefilter.pth`
5. `low5_residual_connector_var_alpha01_norefilter.pth`
6. `beauty_plain_connector_var.pth`
7. `beauty_residual_connector_var_alpha01.pth`

### Checkpoint-to-Claim Mapping

| Claim/Table | Checkpoint(s) |
| --- | --- |
| Cold-10 main table baseline (LLMInit-Var, no connector) | `results/checkpoints/cold10_llminit_var_no_connector.pth` |
| Cold-10 main table plain connector | `results/checkpoints/cold10_plain_connector_var.pth` |
| Cold-10 main table residual connector (`alpha=0.1`) | `results/checkpoints/cold10_residual_connector_var_alpha01.pth` |
| Low-5 appendix table plain connector | `results/checkpoints/low5_plain_connector_var_norefilter.pth` |
| Low-5 appendix table residual connector (`alpha=0.1`) | `results/checkpoints/low5_residual_connector_var_alpha01_norefilter.pth` |
| Fairness connector row on standard Beauty (plain connector) | `results/checkpoints/beauty_plain_connector_var.pth` |
| Fairness connector row on standard Beauty (residual connector, `alpha=0.1`) | `results/checkpoints/beauty_residual_connector_var_alpha01.pth` |

### Mapping Verification Source (All Runs)

To double-check mapping and run metadata, use checkpoint inventory CSVs:

1. Raw inventory generated from logs: `logs_connector/checkpoint_inventory.csv`
2. Curated snapshot for reproducibility: `results/checkpoints/checkpoint_inventory.csv`

The inventory includes `dataset`, `emb_selection`, `use_connector`, `use_residual_connector`, `residual_connector_alpha`, and connector-weight presence, which are the fields used to verify the checkpoint-to-claim mapping above.

The detailed checkpoint notes are also listed in `results/checkpoints/README.md`.

## Acknowledgement
The structure of this repo is built based on [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their great work.
