"""
reconstruct.py
--------------
Reconstructs perceived visual images from pre-decoded fMRI prediction files
using the frozen Stable Diffusion v1.4 pipeline.

Implements the dual-pathway conditioning inference described in:
  Naru, I. (2026). RetinaReplay: Generative Reconstruction of Visual
  Perceptions from Brain fMRI Data via Lightweight MLP Encoders.
  arXiv preprint.

Building on the inference pipeline of Takagi & Nishimoto (CVPR 2023).

HOW IT WORKS
------------
Two pre-decoded prediction arrays (produced by the RetinaReplay training
pipeline) condition Stable Diffusion at different stages:

  scores_latent  (Early Visual Cortex predictions, shape: n_images × 6400)
      → reshaped to (4, 40, 40) → SD VAE Decoder → coarse structural image
      → re-encoded → noisy latent at step T (forward diffusion)

  scores_c  (Ventral Visual Cortex predictions, shape: n_images × 59136)
      → reshaped to (77, 768) → CLIP conditioning for U-Net denoising
      → guides reverse diffusion toward semantic content

GATED ACCESS
------------
Prediction files and SD v1.4 weights are hosted on HuggingFace under
gated access. Request access and set your token before running:

  export RETINAREPLAY_TOKEN=your_hf_token_here

  Request access: https://huggingface.co/datasets/IshantSingh94/RetinaReplay

USAGE
-----
# Reconstruct using best PSM config (MSE + 0.5×CosSim, default):
python reconstruct.py --subject subj01 --gpu 0 --img_start 0 --img_end 99

# Reconstruct using a specific loss configuration:
python reconstruct.py --subject subj01 --gpu 0 --img_start 0 --img_end 99 --loss_config MSE
python reconstruct.py --subject subj01 --gpu 0 --img_start 0 --img_end 99 --loss_config MSE_0.7Cos
python reconstruct.py --subject subj01 --gpu 0 --img_start 0 --img_end 99 --loss_config baseline

# Available loss_config options:
#   MSE          — MSE only
#   MSE_0.3Cos   — MSE + 0.3×CosSim
#   MSE_0.5Cos   — MSE + 0.5×CosSim (best PSM, default)
#   MSE_0.7Cos   — MSE + 0.7×CosSim
#   MSE_KLD      — MSE + 0.1×KLD
#   baseline     — Ridge regression baseline (Takagi & Nishimoto)

# Specific image indices:
python reconstruct.py --subject subj01 --gpu 0 --img_indices 0,5,12,47

# From a directory of ground-truth images:
python reconstruct.py --subject subj01 --gpu 0 --img_dir /path/to/gt_images

OUTPUT
------
Saves to <output_dir>/image-cvpr/<subject>/samples/<img_idx:05d>/:
  {img_idx:05d}_orig.png    — ground truth stimulus
  sample_000.png            — reconstruction sample 1 of 5
  sample_001.png            — reconstruction sample 2 of 5
  ...
  sample_004.png            — reconstruction sample 5 of 5
"""

import argparse
import h5py
import os
import re
import sys
import threading
import time
from contextlib import nullcontext

import numpy as np
import scipy.io
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

# ── NSD Access ────────────────────────────────────────────────────────────────
_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
if _utils_path not in sys.path:
    sys.path.append(_utils_path)
from nsd_access.nsda import NSDAccess

# ── Stable Diffusion (ldm) — local install required ───────────────────────────
_sd_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'codes', 'diffusion_sd1', 'stable-diffusion')
)
if _sd_path not in sys.path:
    sys.path.append(_sd_path)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Gated download
# ══════════════════════════════════════════════════════════════════════════════

HF_REPO_ID   = "IshantSingh94/RetinaReplay"
HF_REPO_TYPE = "dataset"

# Available prediction configurations — matches filenames on HuggingFace
# Each entry: (early_cortex_filename, ventral_cortex_filename)
PREDICTION_CONFIGS = {
    "MSE":          ("early/{s}_early_init_latent_pred_1.0MSE_0.0Cos.npy",
                     "ventral/{s}_ventral_c_pred_1.0MSE_0.0Cos.npy"),
    "MSE_0.3Cos":   ("early/{s}_early_init_latent_pred_1.0MSE_0.3Cos.npy",
                     "ventral/{s}_ventral_c_pred_1.0MSE_0.3Cos.npy"),
    "MSE_0.5Cos":   ("early/{s}_early_init_latent_pred_1.0MSE_0.5Cos.npy",
                     "ventral/{s}_ventral_c_pred_1.0MSE_0.5Cos.npy"),
    "MSE_0.7Cos":   ("early/{s}_early_init_latent_pred_1.0MSE_0.7Cos.npy",
                     "ventral/{s}_ventral_c_pred_1.0MSE_0.7Cos.npy"),
    "MSE_KLD":      ("early/{s}_early_init_latent_pred_KLD.npy",
                     "ventral/{s}_ventral_c_pred_MSE_pt1KLD.npy"),
    "baseline":     ("early/{s}_early_scores_init_latent.npy",
                     "ventral/{s}_ventral_scores_c.npy"),
}

# SD weights — fixed regardless of config
SD_FILES = {
    "sd_ckpt":   "weights/sd-v1-4.ckpt",
    "sd_config": "weights/v1-inference.yaml",
}


def get_prediction_files(subject, loss_config):
    """
    Build the HuggingFace file paths for the chosen loss configuration.
    Returns a dict with keys: scores_latent, scores_c, sd_ckpt, sd_config.
    """
    if loss_config not in PREDICTION_CONFIGS:
        print(
            f"\n  Unknown loss_config '{loss_config}'.\n"
            f"  Available options: {list(PREDICTION_CONFIGS.keys())}\n"
        )
        sys.exit(1)

    early_path, ventral_path = PREDICTION_CONFIGS[loss_config]
    return {
        "scores_latent": early_path.format(s=subject),
        "scores_c":      ventral_path.format(s=subject),
        **SD_FILES,
    }


def get_token():
    """
    Retrieve the HuggingFace access token from the environment.
    Exits with a clear message if not set.
    """
    token = os.environ.get("RETINAREPLAY_TOKEN")
    if not token:
        print(
            "\n  RetinaReplay prediction files are gated.\n"
            "  Request access at:\n"
            "    https://huggingface.co/datasets/IshantSingh94/RetinaReplay\n\n"
            "  Once approved, generate a token at huggingface.co → Settings → Access Tokens\n"
            "  Then run:\n"
            "    export RETINAREPLAY_TOKEN=your_token_here\n"
        )
        sys.exit(1)
    return token


def download_gated_files(file_dict, cache_dir):
    """
    Download prediction .npy files and SD weights from the gated HuggingFace
    repository on first run. Subsequent runs use the local cache.

    Returns a dict of {key: local_path} for all required files.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    token = get_token()
    paths = {}

    for key, hf_path in file_dict.items():
        local_name = os.path.basename(hf_path)
        local_path = os.path.join(cache_dir, local_name)

        if os.path.exists(local_path):
            print(f"  ✓ {local_name} (cached)")
            paths[key] = local_path
            continue

        print(f"  ↓ Downloading {local_name}...")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type=HF_REPO_TYPE,
                token=token,
                local_dir=cache_dir,
            )
            paths[key] = downloaded
            print(f"  ✓ {local_name}")
        except Exception as e:
            print(
                f"\n  Failed to download {hf_path}.\n"
                f"  Error: {e}\n"
                f"  Check your token and access approval at:\n"
                f"    https://huggingface.co/datasets/{HF_REPO_ID}\n"
            )
            sys.exit(1)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Model utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_sd_model(config_path, ckpt_path, gpu, verbose=False):
    """Load frozen Stable Diffusion v1.4 model onto the specified GPU."""
    print(f"Loading SD model from {ckpt_path}")
    config = OmegaConf.load(config_path)
    pl_sd  = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"  Global step: {pl_sd['global_step']}")
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(pl_sd["state_dict"], strict=False)
    if verbose:
        if missing:    print("  Missing keys:",    missing)
        if unexpected: print("  Unexpected keys:", unexpected)
    model.cuda(f"cuda:{gpu}").eval()
    return model


def img_arr_to_tensor(img_arr):
    """
    Convert a NumPy HWC uint8 image to a normalised PyTorch tensor
    in range [-1, 1] with shape (1, 3, 512, 512).
    """
    image  = Image.fromarray(img_arr).convert("RGB").resize((512, 512), Image.LANCZOS)
    arr    = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)[None])
    return 2.0 * tensor - 1.0


def save_image_async(img_array, path):
    """Save a uint8 numpy image to disk in a background thread."""
    Image.fromarray(img_array.astype(np.uint8)).save(path)


# ══════════════════════════════════════════════════════════════════════════════
# Core reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct(
    model, sampler, img_indices,
    scores_latent, scores_c,
    subject, device,
    nsd_dir, mrifeat_dir, output_dir,
    ddim_steps, strength, scale, n_iter,
):
    """
    Run the dual-pathway reconstruction pipeline for a list of test image indices.

    For each image:
      1. scores_latent → reshape (4,40,40) → SD VAE Decoder → coarse image
      2. Coarse image → re-encode → forward diffusion to step T
      3. scores_c     → reshape (77,768)  → CLIP conditioning
      4. DDIM reverse diffusion conditioned on scores_c → n_iter samples saved
    """
    t_enc       = int(strength * ddim_steps)
    base_output = os.path.join(output_dir, f"image-cvpr/{subject}/samples")
    os.makedirs(base_output, exist_ok=True)

    # ── Load NSD test stimulus mapping ────────────────────────────────────────
    expdesign   = scipy.io.loadmat(
        os.path.join(nsd_dir, 'nsddata', 'experiments', 'nsd', 'nsd_expdesign.mat')
    )
    sharedix    = expdesign['sharedix'] - 1    # 0-based shared (test) indices
    nsda        = NSDAccess(nsd_dir)
    sf          = h5py.File(nsda.stimuli_file, 'r')
    sdataset    = sf.get('imgBrick')
    stims_ave   = np.load(os.path.join(mrifeat_dir, subject, f'{subject}_stims_ave.npy'))
    test_idx_list = np.where(
        np.array([0 if s in sharedix else 1 for s in stims_ave]) == 0
    )[0]

    # ── Reconstruction loop ───────────────────────────────────────────────────
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():

                uc = model.get_learned_conditioning([""])   # unconditional embedding

                for img_idx in tqdm(img_indices, desc="Reconstructing"):
                    if img_idx >= len(test_idx_list):
                        print(f"  Index {img_idx} out of range — skipping.")
                        continue

                    folder = os.path.join(base_output, f"{img_idx:05d}")
                    os.makedirs(folder, exist_ok=True)

                    # ── Save ground truth ─────────────────────────────────────
                    te     = test_idx_list[img_idx]
                    idx73k = stims_ave[te]
                    orig   = np.squeeze(sdataset[idx73k, :, :, :]).astype(np.uint8)
                    threading.Thread(
                        target=save_image_async,
                        args=(orig, os.path.join(folder, f"{img_idx:05d}_orig.png"))
                    ).start()

                    # ── Stage 1: Decode init_latent → coarse image ────────────
                    # scores_latent row: (6400,) → (4, 40, 40)
                    lat    = scores_latent[img_idx].reshape(4, 40, 40)
                    imgarr = torch.from_numpy(lat).unsqueeze(0).to(device)
                    x      = model.decode_first_stage(imgarr)
                    x      = torch.clamp((x + 1) / 2, 0, 1)
                    coarse = (255. * rearrange(x[0].cpu().numpy(), 'c h w -> h w c')).astype(np.uint8)

                    # ── Stage 2: Re-encode coarse image → init_lat ────────────
                    init     = img_arr_to_tensor(coarse).to(device)
                    init_lat = model.get_first_stage_encoding(model.encode_first_stage(init))

                    # ── Stage 3: Build CLIP conditioning from scores_c ────────
                    # scores_c row: (59136,) → (77, 768) = CLIP token × embedding dim
                    c = torch.from_numpy(
                        scores_c[img_idx].reshape(77, 768)
                    ).unsqueeze(0).to(device)

                    # ── Stage 4: DDIM — forward noise + conditioned denoising ──
                    for it in range(n_iter):
                        z_enc   = sampler.stochastic_encode(
                            init_lat, torch.tensor([t_enc]).to(device)
                        )
                        samples = sampler.decode(
                            z_enc, c, t_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                        )
                        out = model.decode_first_stage(samples)
                        out = torch.clamp((out + 1) / 2, 0, 1)
                        arr = (255. * rearrange(out[0].cpu().numpy(), 'c h w -> h w c')).astype(np.uint8)
                        threading.Thread(
                            target=save_image_async,
                            args=(arr, os.path.join(folder, f"sample_{it:03d}.png"))
                        ).start()

    sf.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct visual images from fMRI predictions using SD v1.4."
    )
    parser.add_argument("--subject",     required=True, type=str,
                        help="Subject ID (e.g. subj01).")
    parser.add_argument("--gpu",         required=True, type=int,
                        help="GPU index.")
    parser.add_argument(
        "--loss_config",
        type=str,
        default="MSE_0.5Cos",
        choices=list(PREDICTION_CONFIGS.keys()),
        help=(
            "Which prediction configuration to use. "
            "MSE_0.5Cos is the best PSM config from the paper. "
            f"Options: {list(PREDICTION_CONFIGS.keys())} (default: MSE_0.5Cos)"
        )
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. (default: 42)")

    # Image selection — one of three modes
    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument("--img_start",    type=int,
                     help="Start index of image range (use with --img_end).")
    sel.add_argument("--img_indices",  type=str,
                     help="Comma-separated list of image indices, e.g. '0,5,12,47'.")
    sel.add_argument("--img_dir",      type=str,
                     help="Directory of images with zero-padded index filenames.")
    parser.add_argument("--img_end",   type=int, default=None,
                        help="End index (inclusive) for --img_start range.")

    # Paths
    parser.add_argument("--nsd_dir",     type=str, default="../../nsd/",
                        help="Root NSD dataset directory. (default: ../../nsd/)")
    parser.add_argument("--mrifeat_dir", type=str, default="../../mrifeat/",
                        help="Directory of fMRI feature arrays. (default: ../../mrifeat/)")
    parser.add_argument("--output_dir",  type=str, default="../../decoded/",
                        help="Root output directory for reconstructions. (default: ../../decoded/)")
    parser.add_argument("--cache_dir",   type=str, default=".cache/retinareplay",
                        help="Local cache for gated HuggingFace files. (default: .cache/retinareplay)")

    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    seed_everything(args.seed)

    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Build file dict for chosen loss configuration ──────────────────────────
    file_dict = get_prediction_files(args.subject, args.loss_config)
    print(f"\nConfiguration: {args.loss_config}")
    print(f"  Early cortex file: {os.path.basename(file_dict['scores_latent'])}")
    print(f"  Ventral cortex file: {os.path.basename(file_dict['scores_c'])}")

    # ── Download gated files (or use cache) ───────────────────────────────────
    print("\nChecking gated files...")
    gated = download_gated_files(file_dict, args.cache_dir)

    # ── Load prediction arrays ────────────────────────────────────────────────
    print("\nLoading prediction arrays...")
    scores_latent = np.load(gated["scores_latent"])
    scores_c      = np.load(gated["scores_c"])
    print(f"  scores_latent: {scores_latent.shape}")
    print(f"  scores_c:      {scores_c.shape}")

    # ── Load SD model ─────────────────────────────────────────────────────────
    model   = load_sd_model(gated["sd_config"], gated["sd_ckpt"], args.gpu)
    sampler = DDIMSampler(model)

    ddim_steps = 50
    strength   = 0.8
    scale      = 5.0
    n_iter     = 5

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
    print(f"\nDDIM: steps={ddim_steps}, strength={strength}, "
          f"scale={scale}, samples_per_image={n_iter}")

    # ── Build image index list ────────────────────────────────────────────────
    if args.img_indices:
        idxs = list(map(int, args.img_indices.split(',')))
    elif args.img_dir:
        idxs = sorted([
            int(m.group(1))
            for fn in os.listdir(args.img_dir)
            for m in [re.match(r'^(\d+)\.png$', fn)] if m
        ])
    else:
        if args.img_end is None:
            print("Error: --img_end is required when using --img_start.")
            sys.exit(1)
        idxs = list(range(args.img_start, args.img_end + 1))

    print(f"\nProcessing {len(idxs)} images: {idxs[0]} → {idxs[-1]}")

    # ── Run reconstruction ────────────────────────────────────────────────────
    t0 = time.time()
    reconstruct(
        model=model,
        sampler=sampler,
        img_indices=idxs,
        scores_latent=scores_latent,
        scores_c=scores_c,
        subject=args.subject,
        device=device,
        nsd_dir=args.nsd_dir,
        mrifeat_dir=args.mrifeat_dir,
        output_dir=args.output_dir,
        ddim_steps=ddim_steps,
        strength=strength,
        scale=scale,
        n_iter=n_iter,
    )
    elapsed = time.time() - t0
    print(f"\nDone: {len(idxs)} images in {elapsed:.1f}s "
          f"({elapsed / len(idxs):.2f}s/img)")
    print(f"Outputs saved to: {args.output_dir}/image-cvpr/{args.subject}/samples/")


if __name__ == "__main__":
    main()
