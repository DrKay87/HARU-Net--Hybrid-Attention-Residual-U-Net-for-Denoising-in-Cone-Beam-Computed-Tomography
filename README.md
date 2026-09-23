# HARU-Net: Hybrid Attention Residual U-Net for Edge-Preserving Denoising in Cone-Beam Computed Tomography

Official code repository for **HARU-Net**, a deep-learning framework for denoising low-dose cone-beam computed tomography (CBCT) while preserving fine anatomical structures.

**Paper:** Khuram Naveed and Ruben Pauwels, *Biomedical Physics & Engineering Express* (2026).  
**DOI:** 10.1088/2057-1976/ae9c46  
**arXiv:** 2602.22544

## Overview

Low-dose CBCT is desirable for reducing radiation exposure, but lower exposure increases spatially varying noise and can obscure fine anatomical structures. HARU-Net combines residual convolutional learning with hybrid attention to balance noise suppression, structural preservation, and computational efficiency.

The architecture contains three main components:

1. **Residual convolutional encoder/decoder blocks** for stable local feature extraction.
2. **Hybrid Attention Blocks (HABs)** in the skip connections to emphasize salient anatomical features.
3. **Residual hybrid-attention processing at the bottleneck** to model broader contextual interactions.

The network predicts a residual correction and adds it to the input image.

## Published results

The peer-reviewed paper reports **37.52 dB PSNR**, **0.9557 SSIM**, and **0.1084 GMSD** for HARU-Net on the study test data. Please consult the paper for the complete experimental protocol, baselines, statistical interpretation, and computational comparisons.

## Repository structure

```text
.
├── harunet/
│   ├── __init__.py
│   ├── model.py          # HARU-Net, HAB and window-attention implementation
│   ├── data.py           # Paired pickle-array dataset utilities
│   ├── metrics.py        # PSNR and GMSD
│   └── utils.py          # Original utility functions
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   ├── HARUnet_CBCT3D_data_Denoising.ipynb
│   ├── Generate_VisualResults_HARUNet_Paper_Batchwise.ipynb
│   └── Generate_HARUnet_ProcTime_Results_on_consumerGPU.ipynb
├── results/              # Generated outputs (ignored except README)
├── assets/               # Public figures/diagrams for the README
├── data/                 # Local/private data (ignored except README)
├── checkpoints/          # Model weights (ignored except README)
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── .gitignore
```

## Installation

```bash
git clone https://github.com/DrKay87/HARU-Net--Hybrid-Attention-Residual-U-Net-for-Denoising-in-Cone-Beam-Computed-Tomography.git
cd HARU-Net--Hybrid-Attention-Residual-U-Net-for-Denoising-in-Cone-Beam-Computed-Tomography

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA/CPU environment if the default installation does not match your system.

## Data

The original study data are not distributed in this repository. The training workflow expects paired noisy/input and reference/target CBCT patches. The supplied research notebook uses pickled NumPy arrays; `harunet/data.py` preserves this convention for the command-line scripts.

Keep restricted or patient-derived data outside Git and provide local paths at runtime.

## Training

```bash
python scripts/train.py \
  --train-inputs /path/to/train_inputs.pkl \
  --train-targets /path/to/train_targets.pkl \
  --val-inputs /path/to/val_inputs.pkl \
  --val-targets /path/to/val_targets.pkl \
  --epochs 100 \
  --batch-size 8
```

The defaults reproduce the principal model configuration represented in the supplied implementation: base dimension 64, 8 attention heads, window size 16, and bottleneck attention depth 6.

## Evaluation

```bash
python scripts/evaluate.py \
  --test-inputs /path/to/test_inputs.pkl \
  --test-targets /path/to/test_targets.pkl \
  --checkpoint checkpoints/best.pt
```

For full paper-style comparisons and visual-result generation, see the result-generation notebooks described below.

## Inference

For a normalized NumPy image or slice stack:

```bash
python scripts/inference.py \
  --input /path/to/noisy_cbct.npy \
  --checkpoint checkpoints/best.pt \
  --output results/denoised.npy
```

The current architecture uses windowed attention, so image dimensions entering the attention blocks must be compatible with the configured window size and network downsampling.

## Reproducing the paper workflow

The repository retains the original research notebooks because they contain details useful for reproducing and auditing the experimental workflow:

- **`HARUnet_CBCT3D_data_Denoising.ipynb`** — training, validation, testing, checkpointing, and the original experimental workflow.
- **`Generate_VisualResults_HARUNet_Paper_Batchwise.ipynb`** — batchwise quantitative evaluation and generation of visual comparisons used during paper result preparation.
- **`Generate_HARUnet_ProcTime_Results_on_consumerGPU.ipynb`** — inference/runtime benchmarking on a consumer GPU and comparison with other denoising methods.

> **Important:** The notebooks originate from the research environment and may contain machine-specific paths and references to locally stored datasets/checkpoints. Update those paths before execution. Do not commit restricted CBCT data or non-public patient images.

## Using the model in Python

```python
import torch
from harunet import HARU_net

model = HARU_net(
    dim=64,
    hab_heads=8,
    hab_ws=16,
    hab_mlp_ratio=2.0,
    hab_depth=6,
)

x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = model(x)

print(y.shape)
```

## Citation

If you use HARU-Net, please cite:

```bibtex
@article{naveed2026harunet,
  title   = {HARU-Net: Hybrid Attention Residual U-Net for Edge-Preserving Denoising in Cone-Beam Computed Tomography},
  author  = {Naveed, Khuram and Pauwels, Ruben},
  journal = {Biomedical Physics \& Engineering Express},
  year    = {2026},
  doi     = {10.1088/2057-1976/ae9c46}
}
```

## License

This repository is provided under the MIT License for the code included here. Dataset access, third-party implementations, pretrained weights, and publication figures may be subject to separate terms.

## Acknowledgments

This repository accompanies research carried out at Aarhus University. Please refer to the published paper for the complete acknowledgments, experimental details, and references.
