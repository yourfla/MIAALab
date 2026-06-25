# MOSAIC

**Multi-rater Opinion Segmentation with Annotator-Informed Calibration**

MOSAIC is a unified framework for multi-rater medical image segmentation. It
learns directly from a panel of expert annotations and produces outputs that are
both *diversified* across the space of plausible interpretations and
*personalized* to individual annotators, through three gradient-isolated modules:

- **SC-ECRD** — Style-Conditioned Expert-Aware Conditional Refinement Diffusion,
  for attributable and calibrated diversity.
- **EBF** — Evidential Belief Fusion, for a single-pass decomposition of inherent
  and inter-rater uncertainty together with per-annotator predictions.
- **SABR** — Spatially-Aware Boundary Refinement, which concentrates the
  per-annotator correction on the contested boundary regions.

This repository releases the core model code for the LIDC-IDRI and NPC-170
benchmarks.

## Installation

```bash
pip install -r requirements.txt
```

The implementation follows the environment of
[D-Persona](https://github.com/ycwu1997/D-Persona) (PyTorch + SMP ResNet-34
backbone).

## Repository structure

```
MOSAIC/
├── models/                     # core model and metrics
│   ├── mosaic.py               #   MOSAIC (SC-ECRD + EBF + SABR)
│   ├── dpersona.py             #   D-Persona baseline
│   ├── initialize_model.py     #   model factory
│   ├── initialize_optimization.py
│   ├── metrics_set.py          #   GED / Dice_match
│   └── uncertainty_metrics.py  #   NCC / AUDC / DSD / BoundaryDice
├── pionono_models/             # baselines (Pionono, CM, supervised)
├── Probabilistic_Unet_Pytorch/ # Probabilistic U-Net backbone / baseline
├── dataloader/                 # LIDC-IDRI / NPC-170 data loaders
└── configs/                    # dataset configuration
```

## Data preparation

- **LIDC-IDRI** is publicly available from its original source; we use the
  four-rater 2D version under patient-level four-fold cross-validation.
- **NPC-170** is available through the MMIS-2024 Grand Challenge (ACM MM 2024).

Set the dataset path in the corresponding config under `configs/`
(`params_lidc.yaml` for LIDC-IDRI, `params_npc.yaml` for NPC-170).

## Usage

MOSAIC is selected with `--model_name MOSAIC` and trained with a two-stage
protocol. The first stage fits the base segmentation network (U-Net,
prior/posterior encoders and segmentation head) and saves a base checkpoint. The
second stage loads that base checkpoint, keeps the base network frozen, and
trains the three auxiliary modules (SC-ECRD, EBF and SABR) on top of it, so the
modules are optimized without altering the base representation. The example below
uses LIDC-IDRI (`--mask_num 4`, four-fold cross-validation); NPC-170 follows the
same steps with `configs/params_npc.yaml`.

**Stage 0 — base network.** Train the base segmentation network and save its
checkpoint.

```bash
python train.py --config configs/params_lidc.yaml \
    --model_name MOSAIC --stage base \
    --mask_num 4 --save_path ./output/lidc/
```

**Stage 1 — auxiliary modules.** Load the Stage 1 base checkpoint, freeze the
base network, and train SC-ECRD, EBF and SABR.

```bash
python train.py --config configs/params_lidc.yaml \
    --model_name MOSAIC --stage aux \
    --base_ckpt ./output/lidc/ \
    --mask_num 4 --save_path ./output/lidc/
```

**Stage 2 — personalization (optional).** Fine-tune the per-annotator heads.

```bash
python train.py --config configs/params_lidc.yaml \
    --model_name MOSAIC --stage personalize \
    --mask_num 4 --save_path ./output/lidc/
```

**Evaluation.** Report diversity (GED), personalization (Dice_match) and the
uncertainty metrics (NCC, AUDC, DSD, BoundaryDice).

```bash
python evaluate.py --config configs/params_lidc.yaml \
    --model_name MOSAIC --mask_num 4 --save_path ./output/lidc/
```

> Training and evaluation entry scripts are dataset- and environment-specific and
> are not included in this release; the snippets above describe the intended
> workflow and the arguments the model factory in `models/initialize_model.py`
> expects.

## Acknowledgements

The backbone and several baselines build on the public implementations of the
Probabilistic U-Net, Pionono, and D-Persona. We thank the authors for releasing
their code.

## Maintained by

MIAALab.
