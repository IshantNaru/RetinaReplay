# RetinaReplay — Generative Reconstruction of Visual Perceptions from Brain fMRI Data

<p align="center">
  <img src="assets/pipeline_overview.png" alt="RetinaReplay Pipeline" width="600"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Preprint-b31b1b?style=flat&logo=arxiv"/></a>
  <a href="https://huggingface.co/datasets/IshantSingh94/RetinaReplay"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Gated%20Dataset-FFD21F"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey"/></a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch"/>
</p>

---

> **RetinaReplay** is a lightweight fMRI-to-image reconstruction framework that replaces the linear ridge-regression encoders of the [Takagi & Nishimoto (CVPR 2023)](https://arxiv.org/abs/2303.05334) baseline with nonlinear MLP encoders trained with composite loss functions, achieving substantial improvements in both encoder correlation and downstream perceptual reconstruction quality — without the overhead of current state-of-the-art pipelines.

---

## What is fMRI Visual Decoding?

When you look at an image, your brain generates a unique pattern of neural activity — a kind of fingerprint of what you're seeing. RetinaReplay asks: if we can measure that brain activity, can we reconstruct the original image from it?

This is made possible by fMRI — a brain scanning technique that measures blood flow changes across the brain as a proxy for neural activity. The **Natural Scenes Dataset (NSD)** captured these signals from 8 subjects while they viewed 73,000 natural images inside a 7-Tesla MRI scanner — one of the most powerful available. Each image was shown 3 times per subject across 30–40 scanning sessions, producing a rich dataset of brain responses to real-world visual scenes.

The challenge: given only the brain signals, recover what the person was looking at.

---

## The Natural Scenes Dataset (NSD)

| Property | Detail |
|---|---|
| Subjects | 8 (4 full-data: subj01, subj02, subj05, subj07) |
| Scanner | 7-Tesla MRI |
| Images | 73,000 MS-COCO natural scenes, center-cropped to 425×425 |
| Sessions | 30–40 per subject |
| Trials | ~750 per session (each image shown 3 times) |
| Test set | 1,000 shared images seen by all subjects |
| Annotations | 5 COCO captions per image |
| fMRI format | Beta weights at 1.8mm isotropic resolution |

**This project uses Subject 1 only**, restricted to one subject due to computational constraints at the time of the project. Training used 24,980 individual trials; evaluation used 982 averaged test samples. Hardware: NVIDIA RTX 4060 (8GB) for preprocessing and reconstruction, NVIDIA A100 (40GB, Google Colab Pro) and RTX 4060 for MLP training.

NSD data access: [naturalscenesdataset.org](http://naturalscenesdataset.org)

---

## How RetinaReplay Works

The pipeline uses two brain regions that process visual information differently:

- **Early Visual Cortex** — encodes low-level structure: edges, shapes, spatial layout
- **Ventral Visual Cortex** — encodes high-level semantics: objects, categories, meaning

Each region's fMRI signals are mapped to a different part of the Stable Diffusion v1.4 conditioning space:

```
Early Cortex  → MLP1 → init_latent (6400-dim) → SD VAE Decoder → coarse image
Ventral Cortex → MLP2 → c (59136-dim)          → U-Net conditioning → semantic guidance
```

The two streams condition Stable Diffusion at different stages — spatial structure from Early cortex initialises the image, semantic content from Ventral cortex guides the denoising.

---

## Results at a Glance

### Encoder Performance (Held-out Test Set, Subject 1 NSD)

| ROI | Model | Mean Pearson r | Δr vs Baseline |
|---|---|:---:|:---:|
| Early Visual Cortex | Ridge Regression (Takagi & Nishimoto) | 0.239 | — |
| Early Visual Cortex | **MLP (Ours)** | **0.306** | **+28%** |
| Ventral Visual Cortex | Ridge Regression (Takagi & Nishimoto) | 0.304 | — |
| Ventral Visual Cortex | **MLP (Ours)** | **0.767** | **+152%** |

### Downstream Reconstruction Quality (100 Test Images)

| Method | Encoder | Loss | PSM ↑ | SFS ↑ |
|---|---|---|:---:|:---:|
| Takagi & Nishimoto | Ridge | MSE | 0.457 | 0.275 |
| **Ours** | MLP | MSE | 0.542 | **0.359** |
| **Ours** | MLP | MSE + 0.1×KLD | 0.546 | 0.355 |
| **Ours** | MLP | MSE + 0.3×CosSim | 0.540 | 0.355 |
| **Ours** | MLP | MSE + 0.5×CosSim | 0.547 | 0.356 |
| **Ours** | MLP | MSE + 0.7×CosSim | **0.559** | 0.354 |
| **Ours (hybrid)** | Ridge (z) + MLP (c) | Mixed | **0.579** | 0.344 |

> **PSM** = AlexNet Perceptual Similarity Metric · **SFS** = CLIP Semantic Fidelity Score · Higher is better for both.

---

## Qualitative Comparisons

![Qualitative Results Grid 1](assets/results_grid1.png)
*Col 1 (red border): Ground Truth · Col 2: Takagi & Nishimoto baseline · Cols 3–8: RetinaReplay variants*

![Qualitative Results Grid 2](assets/results_grid2.png)

> RetinaReplay reconstructions show **sharper structure**, **improved semantic coherence**, and **reduced checkerboard artifacts** consistently present in the ridge regression baseline.

---

## Method Overview

RetinaReplay builds directly on the dual-pathway conditioning framework of Takagi & Nishimoto, replacing their ridge-regression encoders with four targeted improvements:

| Contribution | Description |
|---|---|
| **MLP Encoders** | Nonlinear feedforward networks replace ridge regression for both Early and Ventral cortex pathways |
| **PCA Compression** | Dimensionality reduction applied prior to each MLP, reducing parameters by 34.2% while preserving reconstruction quality |
| **ELU Activation** | Smooth gradient flow, near-zero mean activations, selected via systematic ablation over 5 activation functions |
| **Composite Loss** | MSE combined with cosine-similarity loss and KL Divergence, targeting both magnitude and directional alignment |

> Full architectural details, ablation tables, and methodology are in the [paper](https://arxiv.org/abs/XXXX.XXXXX).
> Training code is not released. Reproduction is supported via pre-decoded prediction files (see below).

---

## Repository Structure

```
RetinaReplay/
│
├── README.md
├── LICENSE                         ← CC BY-NC-ND 4.0
├── requirements.txt
│
├── test_data/                      ← 100 NSD ground truth test images (Subject 1)
│   ├── 000.png                     ← corresponds to prediction array row 0
│   ├── 001.png
│   ...
│   └── 099.png
│
├── stage1_preprocessing/           ← fMRI & stimulus preprocessing
│   ├── make_subjmri.py             ← ROI beta extraction (Dask-optimised)
│   ├── img2feat_sd.py              ← Image & caption → SD/CLIP embeddings
│   └── make_subjstim.py            ← Subject-specific feature stacking
│
├── stage3_reconstruction/          ← Image reconstruction from predictions
│   └── reconstruct.py             ← Requires gated HuggingFace access
│
├── stage4_evaluation/              ← Perceptual & semantic evaluation
│   └── evaluate.py                ← AlexNet PSM + CLIP SFS scoring
│
└── assets/                         ← Figures and diagrams
```

> **Stage 2 (Training)** is intentionally not included.
> Pre-decoded `.npy` prediction files are available via gated access below.

---

## Reproducing Results

### Step 1 — Clone the Repository

```bash
git clone https://github.com/IshantNaru/RetinaReplay
cd RetinaReplay
```

### Step 2 — Install Dependencies

See `requirements.txt` for the full dependency list. Key steps:

```bash
# Install PyTorch with correct CUDA version for your GPU
# Check your CUDA version with: nvidia-smi
# Example for CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install setuptools==80.10.2
pip install -r requirements.txt

# Install OpenAI CLIP (requires --no-build-isolation)
pip install git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1 --no-build-isolation

# Install NSD Access library
pip install git+https://github.com/tknapen/nsd_access.git

# Clone and install Stable Diffusion v1.4 (ldm)
mkdir -p codes/diffusion_sd1
git clone https://github.com/CompVis/stable-diffusion codes/diffusion_sd1/stable-diffusion
cd codes/diffusion_sd1/stable-diffusion && pip install -e . && cd ../../..

# Clone and install Taming Transformers (required by ldm)
git clone https://github.com/CompVis/taming-transformers codes/diffusion_sd1/taming-transformers
cd codes/diffusion_sd1/taming-transformers && pip install -e . && cd ../../..
```

### Step 3 — Request Access to Prediction Files

Pre-decoded prediction files are hosted on HuggingFace under gated access:

👉 **[Request Access — HuggingFace Dataset](https://huggingface.co/datasets/IshantSingh94/RetinaReplay)**

Provide your name, institution, and intended use. Once approved, generate a token at `huggingface.co → Settings → Access Tokens` and set:

```bash
# Linux / Mac
export RETINAREPLAY_TOKEN=your_hf_token_here

# Windows PowerShell
$env:RETINAREPLAY_TOKEN="your_hf_token_here"
```

### Step 4 — Run Reconstruction

```bash
# Reconstruct first 100 test images (default: MSE + 0.5×CosSim config)
python stage3_reconstruction/reconstruct.py \
  --subject subj01 \
  --gpu 0 \
  --img_start 0 \
  --img_end 99

# Try a different loss configuration
python stage3_reconstruction/reconstruct.py \
  --subject subj01 \
  --gpu 0 \
  --img_start 0 \
  --img_end 99 \
  --loss_config MSE_0.7Cos
```

Ground truth images from `test_data/` are automatically copied alongside each reconstruction for visual comparison.

Output structure:
```
reconstructions/
└── MSE_0.5Cos/
    ├── 00000/
    │   ├── 000_orig.png      ← ground truth
    │   ├── sample_000.png    ← reconstruction 1 of 5
    │   └── ...
    └── ...
```

### Step 5 — Run Evaluation

```bash
python stage4_evaluation/evaluate.py \
  --samples_dir reconstructions/MSE_0.5Cos
```

Outputs a CSV with per-image PSM and SFS scores.

---

## Data Requirements

This project uses the **Natural Scenes Dataset (NSD)**:

> Allen, E.J. et al. "A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence." *Nature Neuroscience*, 25, 116–126 (2022).

NSD requires separate access approval at [naturalscenesdataset.org](http://naturalscenesdataset.org).
Preprocessing scripts in `stage1_preprocessing/` assume the standard NSD directory layout.

---

## Citation

If you use RetinaReplay in your research, please cite:

```bibtex
@article{naru2026retinareplay,
  title     = {RetinaReplay: Generative Reconstruction of Visual Perceptions
               from Brain fMRI Data via Lightweight MLP Encoders},
  author    = {Naru, Ishant},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026}
}
```

---

## Acknowledgements

Thanks to Dr. Kashif Rajpoot and Dr. Mian M. Hamayun at the University of Birmingham for supervision and guidance.
Thanks to the NSD team for making their dataset publicly available.
This work builds on the framework of [Takagi & Nishimoto (CVPR 2023)](https://arxiv.org/abs/2303.05334).

---

## License

This repository is released under [CC BY-NC-ND 4.0](LICENSE).
You may share with attribution. Commercial use and derivative works are not permitted.
Pre-decoded prediction files are subject to additional terms of the gated access agreement.
