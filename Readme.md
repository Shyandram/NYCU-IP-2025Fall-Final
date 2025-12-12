# Multi-Exposure Image Fusion (NYCU Fall 2025)

**NYCU_IP_2025Fall - Final Project**  
**Team 19 — MEF | Shyang-En Weng**

## Overview

This project focuses on multi-exposure image fusion, leveraging and extending the Retinex-MEF framework. Enhancements include optional color-matrix adjustments and gamma correction tweaks for improved image quality and flexibility.

## Repo Layout

- Code lives in [RetinexMEF/](RetinexMEF)
	- Inference: [RetinexMEF/test.py](RetinexMEF/test.py)
	- Training: [RetinexMEF/train.py](RetinexMEF/train.py)
	- Evaluation: [RetinexMEF/eval_mef.py](RetinexMEF/eval_mef.py)
	- Pretrained: [RetinexMEF/model/ckpt.pth](RetinexMEF/model/ckpt.pth)
	- Samples: [RetinexMEF/test_case/](RetinexMEF/test_case)

## Setup
Follow the setup as RetinexMEF
```powershell
pip install -r RetinexMEF/requirements.txt
```

## Run Inference (Quick Start)

1) Open [RetinexMEF/test.py](RetinexMEF/test.py) and set:
	 - `use_paired_dataset = False`
	 - `path_model = r"model/ckpt.pth"`
	 - `path_img1 = r"test_case/under"`, `path_img2 = r"test_case/over"`

2) Run:

```powershell
cd RetinexMEF
python test.py
```

Tip: You can toggle `use_color_matrix`, `use_L_net_color`, or `use_adaptive_gamma_correction` if you want to try the modification

## Train (Optional)

Prepare SICE training patches as described in [RetinexMEF/README.md](RetinexMEF/README.md), then:

Before running the training script, you can modify the following options in `train.py` to suit your needs:

use_color_matrix = True  # Set to True to enable color matrix prediction in fusionnet (from img1+img2)
use_L_net_color = True  # Set to True to enable color matrix prediction in L_net (CNN-based)


Tip: You can toggle `use_color_matrix`, `use_L_net_color`, or if you want to try the modification


```powershell
cd RetinexMEF
python train.py
```

Checkpoints/logs are saved under `exp/<timestamp>/`. To avoid Weights & Biases logging, set `use_wandb=False` in [RetinexMEF/train.py](RetinexMEF/train.py) or export `WANDB_API_KEY`.

## Evaluate (Optional)

Requires `pyiqa` and `kornia` in addition to the requirements file.

```powershell
cd RetinexMEF
pip install pyiqa kornia
python eval_mef.py `
	--inputs_root data/SICE2_sorted `
	--gt_root data/SICE2_Label `
	--fused_root results/c2wom `
	--csv_out exp/eval/c2wom.csv
```

<!-- ## Links

- Presentation (YouTube): https://youtu.be/Uc15KUQ8kTs -->

## Acknowledgement

This project builds upon the excellent work of [Retinex-MEF](https://github.com/HaowenBai/Retinex-MEF). It is developed solely for educational purposes as part of a course project. All credit for the original code belongs to its respective authors.
