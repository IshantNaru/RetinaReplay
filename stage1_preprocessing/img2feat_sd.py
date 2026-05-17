"""
img2feat_sd.py
--------------
Converts NSD ground-truth images and their COCO text annotations into latent
space embeddings using the frozen Stable Diffusion v1.4 encoder and CLIP text
encoder respectively.

  init_latent : SD v1.4 VAE encoder output, flattened → shape (6400,)
  c           : CLIP text encoder output, mean-pooled across 5 captions → shape (59136,)

Adapted from Takagi & Nishimoto (CVPR 2023) with the following improvements:
  - Resumable: skips images whose embeddings are already saved
  - Subject-specific stimulus filtering via NSD stimulus info CSV
  - All paths configurable via CLI — no hardcoded directories

USAGE
-----
python img2feat_sd.py --gpu 0 --subject subj01

# Custom paths:
python img2feat_sd.py --gpu 0 --subject subj01 \\
    --nsd_dir /data/nsd \\
    --output_dir /data/nsdfeat \\
    --sd_ckpt /models/sd-v1-4.ckpt

NOTE
----
Requires Stable Diffusion v1.4 (ldm) to be installed locally.
See requirements.txt and README.md for setup instructions.
"""

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import torch
from einops import repeat
from nsd_access import NSDAccess
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

# ── Stable Diffusion (ldm) — local install required ───────────────────────────
_sd_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'codes', 'diffusion_sd1', 'stable-diffusion')
)
sys.path.append(_sd_path)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SD latent and CLIP text embeddings from NSD stimuli."
    )
    parser.add_argument(
        "--gpu", required=True, type=int,
        help="GPU index to use."
    )
    parser.add_argument(
        "--subject", required=True, type=str,
        help="Subject ID (e.g. subj01). Used to filter subject-specific stimuli."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. (default: 42)"
    )
    parser.add_argument(
        "--nsd_dir", type=str, default="../../nsd/",
        help="Path to root NSD dataset directory. (default: ../../nsd/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../../nsdfeat/",
        help="Root output directory for embeddings. (default: ../../nsdfeat/)"
    )
    parser.add_argument(
        "--sd_config", type=str,
        default="../codes/diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        help="Path to SD v1.4 config YAML."
    )
    parser.add_argument(
        "--sd_ckpt", type=str,
        default="../codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="Path to SD v1.4 checkpoint (.ckpt). See README for download instructions."
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Which COCO split to process: train2017 or val2017. (default: train)"
    )
    return parser.parse_args()


def get_subject_stimulus_indices(nsd_dir, subject, split):
    """
    Returns stimulus indices for a given subject and COCO split,
    filtered to images not yet processed (resumable).

    NSD stimulus info CSV maps each of the 73K images to subject
    presentation flags and COCO split labels.
    """
    csv_path = os.path.join(
        nsd_dir, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.csv'
    )
    df = pd.read_csv(csv_path)
    subj_col  = subject.replace('subj', 'subject')   # subj01 → subject1
    coco_split = f"{split}2017"

    indices = df[
        (df[subj_col] == 1) & (df['cocoSplit'] == coco_split)
    ].index.tolist()
    return sorted(indices)


def load_model(config_path, ckpt_path, gpu):
    """Load frozen SD v1.4 model onto specified GPU."""
    print(f"Loading SD model from {ckpt_path}")
    config = OmegaConf.load(config_path)
    pl_sd  = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"  Global step: {pl_sd['global_step']}")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda(f"cuda:{gpu}").eval()
    return model, config


def img_to_tensor(img_arr, resolution=320):
    """
    Convert a NumPy HWC uint8 image to a normalised PyTorch tensor
    in range [-1, 1] with shape (1, 3, H, W).
    """
    image = Image.fromarray(img_arr).convert("RGB")
    image = image.resize((resolution, resolution), resample=PIL.Image.LANCZOS)
    arr   = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr[None].transpose(0, 3, 1, 2))
    return 2.0 * tensor - 1.0


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(args.gpu)
    seed_everything(args.seed)

    # ── Output directories ────────────────────────────────────────────────────
    latent_dir = os.path.join(args.output_dir, 'init_latent')
    c_dir      = os.path.join(args.output_dir, 'c')
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(c_dir,      exist_ok=True)

    # ── Already-processed files (for resumable runs) ──────────────────────────
    processed = {int(f.stem) for f in Path(c_dir).iterdir() if f.is_file()}
    print(f"Found {len(processed)} already-processed images — will skip these.")

    # ── Subject-specific stimulus indices ─────────────────────────────────────
    all_indices = get_subject_stimulus_indices(args.nsd_dir, args.subject, args.split)
    todo        = [i for i in all_indices if i not in processed]
    print(f"Subject {args.subject} | split={args.split}2017 | "
          f"total={len(all_indices)} | remaining={len(todo)}")

    if not todo:
        print("All images already processed. Nothing to do.")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model, _ = load_model(args.sd_config, args.sd_ckpt, args.gpu)
    model    = model.to(device)

    ddim_steps = 50
    ddim_eta   = 0.0
    strength   = 0.8
    scale      = 5.0

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    t_enc = int(strength * ddim_steps)
    print(f"DDIM: steps={ddim_steps}, strength={strength}, t_enc={t_enc}")

    nsda            = NSDAccess(args.nsd_dir)
    precision_scope = autocast

    # ── Main extraction loop ──────────────────────────────────────────────────
    for s in tqdm(todo, desc="Extracting embeddings"):
        print(f"\nImage {s:06d}")

        # Load image
        img        = nsda.read_images(s)
        init_image = img_to_tensor(img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=1)

        # Encode image into SD latent space → init_latent, shape (4, 40, 40)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():

                    uc = model.get_learned_conditioning([""])

                    # Load all COCO captions for this image (typically 5)
                    captions_info = nsda.read_image_coco_info([s], info_type='captions')
                    prompts = [p['caption'] for p in captions_info]
                    print(f"  Captions ({len(prompts)}): {prompts[0][:60]}...")

                    # Encode all captions and mean-pool into a single conditioning vector.
                    # This creates a consensus semantic embedding across all available
                    # annotations rather than conditioning on any single caption,
                    # which reduces sensitivity to individual caption phrasing.
                    c = model.get_learned_conditioning(prompts).mean(axis=0).unsqueeze(0)

        # Move to CPU, detach from computation graph, flatten to 1D
        # init_latent: (1, 4, 40, 40) → (6400,)
        # c:           (1, 77, 768)   → (59136,)
        init_latent_np = init_latent.cpu().detach().numpy().flatten()
        c_np           = c.cpu().detach().numpy().flatten()

        print(f"  init_latent shape: {init_latent_np.shape} | c shape: {c_np.shape}")

        np.save(os.path.join(latent_dir, f'{s:06d}.npy'), init_latent_np)
        np.save(os.path.join(c_dir,      f'{s:06d}.npy'), c_np)

    print(f"\nDone. Embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
