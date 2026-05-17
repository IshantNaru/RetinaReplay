"""
make_subjstim.py
----------------
Stacks per-image embedding files (.npy) into subject-specific training and
test arrays aligned with the fMRI trial order.

Runs after img2feat_sd.py. Produces the final (n_stimuli × feature_dim)
arrays consumed by the MLP training pipeline.

USAGE
-----
# Process init_latent embeddings (SD VAE encoder output):
python make_subjstim.py --subject subj01 --featname init_latent --use_stim each
python make_subjstim.py --subject subj01 --featname init_latent --use_stim ave

# Process c embeddings (CLIP text encoder output):
python make_subjstim.py --subject subj01 --featname c --use_stim each
python make_subjstim.py --subject subj01 --featname c --use_stim ave

  each : uses individual trial presentations (training data)
  ave  : uses averaged presentations across repetitions (test data)

OUTPUT
------
Saves to <nsdfeat_dir>/subjfeat/:
  {subject}_{use_stim}_{featname}_tr.npy  — training split
  {subject}_{use_stim}_{featname}_te.npy  — test split

Also saves:
  <mrifeat_dir>/{subject}/{subject}_stims_tridx.npy  — train/test index array
"""

import argparse
import os

import numpy as np
import scipy.io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stack per-image embeddings into subject-specific train/test arrays."
    )
    parser.add_argument(
        "--subject", type=str, required=True,
        help="Subject ID (e.g. subj01). Full-data subjects: subj01, subj02, subj05, subj07."
    )
    parser.add_argument(
        "--featname", type=str, required=True, choices=["init_latent", "c"],
        help="Embedding type to stack: 'init_latent' (SD VAE) or 'c' (CLIP text)."
    )
    parser.add_argument(
        "--use_stim", type=str, required=True, choices=["each", "ave"],
        help="'each' for individual trial presentations, 'ave' for stimulus-averaged."
    )
    parser.add_argument(
        "--nsd_dir", type=str, default="../../nsd/",
        help="Path to root NSD dataset directory. (default: ../../nsd/)"
    )
    parser.add_argument(
        "--mrifeat_dir", type=str, default="../../mrifeat/",
        help="Directory containing fMRI feature arrays from make_subjmri.py. (default: ../../mrifeat/)"
    )
    parser.add_argument(
        "--nsdfeat_dir", type=str, default="../../nsdfeat/",
        help="Directory containing per-image embedding .npy files. (default: ../../nsdfeat/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    feat_dir  = os.path.join(args.nsdfeat_dir, args.featname)
    save_dir  = os.path.join(args.nsdfeat_dir, 'subjfeat')
    stim_dir  = os.path.join(args.mrifeat_dir, args.subject)
    os.makedirs(save_dir, exist_ok=True)

    # ── Load NSD experiment design ────────────────────────────────────────────
    expdesign_path = os.path.join(
        args.nsd_dir, 'nsddata', 'experiments', 'nsd', 'nsd_expdesign.mat'
    )
    nsd_expdesign = scipy.io.loadmat(expdesign_path)

    # NSD indices are 1-based — subtract 1 for 0-based Python indexing
    sharedix = nsd_expdesign['sharedix'] - 1   # shared stimuli = test set

    # ── Load stimulus index array ─────────────────────────────────────────────
    stim_file = (
        f'{args.subject}_stims_ave.npy' if args.use_stim == 'ave'
        else f'{args.subject}_stims.npy'
    )
    stims = np.load(os.path.join(stim_dir, stim_file))
    print(f"Loaded {len(stims)} stimulus indices from {stim_file}")

    # ── Stack embeddings in trial order ──────────────────────────────────────
    feats    = []
    tr_idx   = np.zeros(len(stims), dtype=np.int8)  # 0=test, 1=train

    for idx, s in tqdm(enumerate(stims), total=len(stims), desc="Stacking embeddings"):
        tr_idx[idx] = 0 if s in sharedix else 1
        feat = np.load(os.path.join(feat_dir, f'{int(s):06d}.npy'))
        feats.append(feat)

    feats = np.stack(feats)
    print(f"Stacked array shape: {feats.shape}")

    # ── Train / Test split ────────────────────────────────────────────────────
    feats_tr = feats[tr_idx == 1, :]
    feats_te = feats[tr_idx == 0, :]
    print(f"Train: {feats_tr.shape} | Test: {feats_te.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    tridx_path = os.path.join(stim_dir, f'{args.subject}_stims_tridx.npy')
    np.save(tridx_path, tr_idx)

    tr_path = os.path.join(save_dir, f'{args.subject}_{args.use_stim}_{args.featname}_tr.npy')
    te_path = os.path.join(save_dir, f'{args.subject}_{args.use_stim}_{args.featname}_te.npy')
    np.save(tr_path, feats_tr)
    np.save(te_path, feats_te)

    print(f"\nSaved:")
    print(f"  {tr_path}")
    print(f"  {te_path}")
    print(f"  {tridx_path}")


if __name__ == "__main__":
    main()
