"""
evaluate.py
-----------
Computes Perceptual Similarity Metric (PSM) and Semantic Fidelity Score (SFS)
for RetinaReplay reconstructions against NSD ground-truth stimuli.

Evaluation protocol follows Takagi & Nishimoto (CVPR 2023), independently
reconstructed from their paper's methodology section and supplement.

METRICS
-------
PSM — Perceptual Similarity Metric
    AlexNet (ImageNet) feature correlations at post-ReLU conv layers 2, 5, 7
    (PyTorch indices: features.1, features.4, features.7).
    Captures low-to-mid level perceptual similarity (edges, textures, parts).

SFS — Semantic Fidelity Score
    CLIP ViT-L/14 feature correlations at transformer layers 7, 12, and final
    (PyTorch indices: resblocks.6, resblocks.11, ln_post).
    Captures high-level semantic alignment between reconstruction and ground truth.

Both metrics use Pearson r between flattened feature vectors, averaged across
the three layers specified for each model.

SELECTION
---------
For each test image, N reconstructions are generated via stochastic DDIM
sampling. The best-scoring sample per metric is selected independently
(best-of-N), and scores are averaged across all evaluated images.

This matches the selection protocol of Takagi & Nishimoto exactly.

USAGE
-----
# Evaluate all reconstructions in a directory:
python evaluate.py --samples_dir ../../decoded/image-cvpr/subj01/samples

# Custom number of samples per image and output path:
python evaluate.py \\
    --samples_dir ../../decoded/image-cvpr/subj01/samples \\
    --n_samples 5 \\
    --output_csv ../../results/subj01_scores.csv

OUTPUT
------
CSV file with columns:
  image_idx   — test image index
  best_clip   — filename of best sample by SFS
  sfs_score   — CLIP semantic fidelity score (best sample)
  best_alex   — filename of best sample by PSM
  psm_score   — AlexNet perceptual similarity score (best sample)

Final row: mean SFS and PSM across all evaluated images.
"""

import argparse
import os

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr
from torchvision import models, transforms
from tqdm import tqdm


# ── Layer configuration ───────────────────────────────────────────────────────

# CLIP ViT-L/14: layers 7, 12, and final (0-indexed → resblocks 6, 11, ln_post)
# As specified in Takagi & Nishimoto supplement.
CLIP_LAYERS = [
    "visual.transformer.resblocks.6",    # transformer layer 7
    "visual.transformer.resblocks.11",   # transformer layer 12
    "visual.ln_post",                    # final layer norm
]

# AlexNet: post-ReLU activations after conv layers 1, 2, 3
# PyTorch features indices 1, 4, 7 correspond to paper's "layers 2, 5, 7"
# As specified in Takagi & Nishimoto supplement.
ALEX_LAYERS = [
    "features.1",   # post-ReLU conv1 → paper layer 2
    "features.4",   # post-ReLU conv2 → paper layer 5
    "features.7",   # post-ReLU conv3 → paper layer 7
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(device):
    """
    Load CLIP ViT-L/14 and AlexNet (ImageNet weights), attach forward hooks
    to extract intermediate layer features, and return feature store dicts.
    """
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()

    alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()

    clip_feats = {}
    alex_feats  = {}

    def make_hook(store, name):
        def hook(module, input, output):
            if output.ndim == 4:
                # Conv feature map (B, C, H, W) → global avg pool → (B, C)
                store[name] = output.mean(dim=(2, 3)).cpu().numpy()
            elif output.ndim == 3:
                # Transformer sequence (B, tokens, D) → pool tokens → (B, D)
                store[name] = output.mean(dim=1).cpu().numpy()
            else:
                # Linear output (B, D)
                store[name] = output.cpu().numpy()
        return hook

    named_modules = dict(clip_model.named_modules())
    for name in CLIP_LAYERS:
        named_modules[name].register_forward_hook(make_hook(clip_feats, name))

    named_modules_alex = dict(alex.named_modules())
    for name in ALEX_LAYERS:
        named_modules_alex[name].register_forward_hook(make_hook(alex_feats, name))

    return clip_model, clip_preprocess, alex, clip_feats, alex_feats


# ── Image preprocessing ───────────────────────────────────────────────────────

def build_alex_preprocess():
    """Standard ImageNet preprocessing for AlexNet."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_image(path, clip_preprocess, alex_preprocess, device, for_clip=True):
    img = Image.open(path).convert("RGB")
    if for_clip:
        return clip_preprocess(img).unsqueeze(0).to(device)
    return alex_preprocess(img).unsqueeze(0).to(device)


# ── Scoring ───────────────────────────────────────────────────────────────────

def extract_features(img_path, clip_model, clip_preprocess,
                     alex, alex_preprocess,
                     clip_feats, alex_feats, device):
    """
    Run a single image through both models and return copies of the
    extracted feature dicts. Clears stores before each forward pass.
    """
    clip_feats.clear()
    with torch.no_grad():
        clip_model.encode_image(
            load_image(img_path, clip_preprocess, alex_preprocess, device, for_clip=True)
        )
    cf = {k: v.copy() for k, v in clip_feats.items()}

    alex_feats.clear()
    with torch.no_grad():
        alex(
            load_image(img_path, clip_preprocess, alex_preprocess, device, for_clip=False)
        )
    af = {k: v.copy() for k, v in alex_feats.items()}

    return cf, af


def pearson_mean(gt_feats, pred_feats, layers):
    """
    Compute mean Pearson r across specified layers between ground-truth
    and predicted feature vectors.
    """
    return float(np.mean([
        pearsonr(gt_feats[l].flatten(), pred_feats[l].flatten())[0]
        for l in layers
    ]))


def score_image(gt_path, sample_paths,
                clip_model, clip_preprocess, alex, alex_preprocess,
                clip_feats, alex_feats, device):
    """
    Score all samples for one test image against its ground truth.
    Returns lists of SFS and PSM scores, one per sample.
    """
    # Extract ground-truth features once
    gt_clip, gt_alex = extract_features(
        gt_path, clip_model, clip_preprocess,
        alex, alex_preprocess, clip_feats, alex_feats, device
    )

    sfs_scores = []
    psm_scores = []

    for sp in sample_paths:
        pred_clip, pred_alex = extract_features(
            sp, clip_model, clip_preprocess,
            alex, alex_preprocess, clip_feats, alex_feats, device
        )
        sfs_scores.append(pearson_mean(gt_clip, pred_clip, CLIP_LAYERS))
        psm_scores.append(pearson_mean(gt_alex, pred_alex, ALEX_LAYERS))

    return sfs_scores, psm_scores


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RetinaReplay reconstructions using PSM (AlexNet) "
            "and SFS (CLIP) metrics following Takagi & Nishimoto (CVPR 2023)."
        )
    )
    parser.add_argument(
        "--samples_dir", required=True, type=str,
        help=(
            "Path to samples directory produced by reconstruct.py. "
            "Expected structure: samples/<img_idx:05d>/<img_idx:05d>_orig.png "
            "and samples/<img_idx:05d>/sample_000.png ... sample_00N.png"
        )
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of reconstruction samples per image. Must match --n_iter used in reconstruct.py. (default: 5)"
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help=(
            "Path to save results CSV. "
            "Defaults to <samples_dir>/<dirname>_scores.csv"
        )
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda', 'cuda:N', or 'cpu'. (default: cuda if available)"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Samples directory: {args.samples_dir}")
    print(f"Samples per image: {args.n_samples}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading CLIP ViT-L/14 and AlexNet...")
    clip_model, clip_preprocess, alex, clip_feats, alex_feats = load_models(device)
    alex_preprocess = build_alex_preprocess()
    print("Models loaded.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    records = []
    image_dirs = sorted([
        d for d in os.listdir(args.samples_dir)
        if os.path.isdir(os.path.join(args.samples_dir, d))
    ])

    for d in tqdm(image_dirs, desc="Evaluating images"):
        folder = os.path.join(args.samples_dir, d)

        try:
            img_idx = int(d)
        except ValueError:
            continue

        gt_path = os.path.join(folder, f"{img_idx:05d}_orig.png")
        if not os.path.isfile(gt_path):
            print(f"  Warning: ground truth not found for index {img_idx} — skipping.")
            continue

        # Collect available sample files
        sample_paths = [
            os.path.join(folder, f"sample_{i:03d}.png")
            for i in range(args.n_samples)
            if os.path.isfile(os.path.join(folder, f"sample_{i:03d}.png"))
        ]
        if not sample_paths:
            print(f"  Warning: no samples found for index {img_idx} — skipping.")
            continue

        sfs_scores, psm_scores = score_image(
            gt_path, sample_paths,
            clip_model, clip_preprocess, alex, alex_preprocess,
            clip_feats, alex_feats, device
        )

        best_sfs_idx = int(np.argmax(sfs_scores))
        best_psm_idx = int(np.argmax(psm_scores))

        records.append({
            "image_idx": img_idx,
            "best_clip": os.path.basename(sample_paths[best_sfs_idx]),
            "sfs_score": round(sfs_scores[best_sfs_idx], 4),
            "best_alex": os.path.basename(sample_paths[best_psm_idx]),
            "psm_score": round(psm_scores[best_psm_idx], 4),
        })

    if not records:
        print("No valid images evaluated. Check your samples directory structure.")
        return

    # ── Build results DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(records, columns=[
        "image_idx", "best_clip", "sfs_score", "best_alex", "psm_score"
    ])

    mean_sfs = df["sfs_score"].mean()
    mean_psm = df["psm_score"].mean()

    avg_row = pd.DataFrame([{
        "image_idx": "Mean",
        "best_clip": "",
        "sfs_score": round(mean_sfs, 4),
        "best_alex": "",
        "psm_score": round(mean_psm, 4),
    }])
    df = pd.concat([df, avg_row], ignore_index=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output_csv:
        csv_path = args.output_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    else:
        base     = os.path.basename(os.path.normpath(args.samples_dir))
        csv_path = os.path.join(args.samples_dir, f"{base}_scores.csv")

    df.to_csv(csv_path, index=False)

    print(f"\n{'─'*50}")
    print(f"Evaluated {len(records)} images")
    print(f"  Mean SFS (CLIP):    {mean_sfs:.4f}")
    print(f"  Mean PSM (AlexNet): {mean_psm:.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
