"""
make_subjmri.py
---------------
Extracts ROI-based fMRI beta weights from NSD HDF5 files and saves them as
numpy arrays for training and testing.

Adapted from Takagi & Nishimoto (CVPR 2023) with the following improvements:
  - Memory-efficient batch loading using Dask arrays
  - 4D → 2D reshape before chunking to avoid Dask shape errors
  - Per-session try/except with graceful skip on load failure
  - Per-ROI memory cleanup after each save
  - Configurable session count and batch size via CLI

USAGE
-----
# Full dataset (37 sessions, recommended):
python make_subjmri.py --subject subj01

# Specific session count or batch size:
python make_subjmri.py --subject subj01 --num_sessions 37 --batch_size 6

# Custom NSD data location:
python make_subjmri.py --subject subj01 --nsd_dir /path/to/nsd

OUTPUT
------
Saves to <output_dir>/<subject>/:
  {subject}_{roi}_betas_tr.npy      — individual trial betas, training set
  {subject}_{roi}_betas_te.npy      — individual trial betas, test set
  {subject}_{roi}_betas_ave_tr.npy  — stimulus-averaged betas, training set
  {subject}_{roi}_betas_ave_te.npy  — stimulus-averaged betas, test set
  {subject}_stims.npy               — all stimulus indices (73K ID, 0-based)
  {subject}_stims_ave.npy           — unique stimulus indices (0-based)
"""

import argparse
import os
import numpy as np
import pandas as pd
import dask.array as da
import scipy.io
from nsd_access import NSDAccess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ROI beta weights from NSD fMRI data."
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID. Full-data subjects: subj01, subj02, subj05, subj07."
    )
    parser.add_argument(
        "--nsd_dir",
        type=str,
        default="../../nsd/",
        help="Path to the root NSD dataset directory. (default: ../../nsd/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../mrifeat/",
        help="Root directory for saving output numpy arrays. (default: ../../mrifeat/)"
    )
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=37,
        help="Number of fMRI sessions to process. Full subjects have 37. (default: 37)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Number of sessions to load per batch. Reduce if running out of RAM. (default: 3)"
    )
    return parser.parse_args()


def load_session_betas(nsda, subject, session_idx):
    """
    Load beta weights for a single fMRI session.
    Returns None on failure so the caller can skip gracefully.
    """
    print(f"  Loading session {session_idx}...")
    try:
        betas = nsda.read_betas(
            subject=subject,
            session_index=session_idx,
            trial_index=[],              # empty = load all trials in session
            data_type='betas_fithrf_GLMdenoise_RR',
            data_format='func1pt8mm'
        )
        if betas is None or len(betas) == 0:
            print(f"  Warning: Session {session_idx} returned empty data — skipping.")
            return None
        print(f"  Session {session_idx} loaded. Shape: {betas.shape}")
        return betas
    except Exception as e:
        print(f"  Error loading session {session_idx}: {e} — skipping.")
        return None


def build_dask_array(nsda, subject, num_sessions, batch_size):
    """
    Load all sessions in memory-efficient batches and concatenate into a
    single Dask array of shape (total_trials, total_voxels).

    Sessions are reshaped from 4D (n_trials, X, Y, Z) to 2D (n_trials, voxels)
    before Dask chunking to avoid shape mismatch errors.
    """
    all_batches = []

    for batch_start in range(1, num_sessions + 1, batch_size):
        batch_end = min(batch_start + batch_size, num_sessions + 1)
        print(f"\nBatch: sessions {batch_start} → {batch_end - 1}")

        batch_arrays = []
        for session_idx in range(batch_start, batch_end):
            data = load_session_betas(nsda, subject, session_idx)
            if data is not None:
                # Reshape 4D (n_trials, X, Y, Z) → 2D (n_trials, voxels)
                data_2d = data.reshape(data.shape[0], -1)
                chunk_size = max(1, data_2d.shape[0] // 4)
                da_sess = da.from_array(data_2d, chunks=(chunk_size, data_2d.shape[1]))
                batch_arrays.append(da_sess)
                print(f"  Converted to Dask: shape={data_2d.shape}, chunks={chunk_size}")

        if not batch_arrays:
            print(f"  No valid sessions in this batch — skipping.")
            continue

        batch_concat = (
            batch_arrays[0] if len(batch_arrays) == 1
            else da.concatenate(batch_arrays, axis=0)
        )
        all_batches.append(batch_concat)
        print(f"  Batch concatenated shape: {batch_concat.shape}")
        del batch_arrays

    if not all_batches:
        raise ValueError("No valid beta data found across all sessions.")

    full_array = (
        all_batches[0] if len(all_batches) == 1
        else da.concatenate(all_batches, axis=0)
    )
    print(f"\nFull Dask array shape: {full_array.shape}")
    return full_array


def main():
    args = parse_args()
    subject = args.subject
    savedir = os.path.join(args.output_dir, subject)
    os.makedirs(savedir, exist_ok=True)

    # ── Load NSD experiment design ────────────────────────────────────────────
    nsda = NSDAccess(args.nsd_dir)
    expdesign_path = os.path.join(
        args.nsd_dir, 'nsddata', 'experiments', 'nsd', 'nsd_expdesign.mat'
    )
    nsd_expdesign = scipy.io.loadmat(expdesign_path)

    # NSD uses 1-based indexing throughout — subtract 1 for 0-based Python indexing
    sharedix = nsd_expdesign['sharedix'] - 1   # stimuli shared across all subjects (test set)

    # ── Load behavioral data to get stimulus presentation order ───────────────
    print(f"\nLoading behavioral data for {subject}...")
    behs = pd.DataFrame()
    for i in range(1, args.num_sessions + 1):
        beh = nsda.read_behavior(subject=subject, session_index=i)
        behs = pd.concat((behs, beh))

    # 73KID is 1-based — subtract 1 for 0-based indexing
    stims_all    = behs['73KID'] - 1             # all trials, preserving repetitions
    stims_unique = behs['73KID'].unique() - 1    # unique stimuli only

    # Save stimulus index arrays (skip if already exist)
    stims_path = os.path.join(savedir, f'{subject}_stims.npy')
    if not os.path.exists(stims_path):
        np.save(stims_path, stims_all)
        np.save(os.path.join(savedir, f'{subject}_stims_ave.npy'), stims_unique)
        print(f"Saved stimulus index arrays to {savedir}")

    # ── Build full Dask array across all sessions ─────────────────────────────
    print(f"\nBuilding Dask array for {args.num_sessions} sessions "
          f"(batch_size={args.batch_size})...")
    betas_all_da = build_dask_array(nsda, subject, args.num_sessions, args.batch_size)

    # ── Process each ROI ──────────────────────────────────────────────────────
    atlas = nsda.read_atlas_results(
        subject=subject, atlas='streams', data_format='func1pt8mm'
    )

    for roi, val in atlas[1].items():
        if val == 0:
            continue
        print(f"\n{'─'*60}")
        print(f"ROI: {roi}  (atlas value={val})")

        # Build voxel mask for this ROI
        atlas_flat = atlas[0].transpose([2, 1, 0]).flatten()
        roi_mask   = atlas_flat == val

        # Lazy Dask slice → compute into memory for this ROI only
        print("  Extracting ROI voxels...")
        betas_roi = betas_all_da[:, roi_mask].compute()
        print(f"  ROI shape: {betas_roi.shape}")

        # Stimulus-averaged betas (average across 3 repetitions per stimulus)
        print("  Computing stimulus averages...")
        betas_roi_ave = []
        for stim in stims_unique:
            stim_trials = np.where(stims_all == stim)[0]
            if len(stim_trials) > 0:
                betas_roi_ave.append(np.mean(betas_roi[stim_trials, :], axis=0))
            else:
                print(f"  Warning: No trials found for stimulus {stim}")
        if not betas_roi_ave:
            print(f"  No valid averages for ROI {roi} — skipping.")
            continue
        betas_roi_ave = np.stack(betas_roi_ave)

        # ── Train / Test split ────────────────────────────────────────────────
        # Shared stimuli (in sharedix) = test set; all others = training set

        # Per-trial split
        betas_tr = np.stack([betas_roi[i] for i, s in enumerate(stims_all) if s not in sharedix])
        betas_te = np.stack([betas_roi[i] for i, s in enumerate(stims_all) if s in sharedix])

        # Averaged split
        betas_ave_tr = np.stack([betas_roi_ave[i] for i, s in enumerate(stims_unique) if s not in sharedix])
        betas_ave_te = np.stack([betas_roi_ave[i] for i, s in enumerate(stims_unique) if s in sharedix])

        # ── Save ──────────────────────────────────────────────────────────────
        saves = {
            f'{subject}_{roi}_betas_tr.npy':     betas_tr,
            f'{subject}_{roi}_betas_te.npy':     betas_te,
            f'{subject}_{roi}_betas_ave_tr.npy': betas_ave_tr,
            f'{subject}_{roi}_betas_ave_te.npy': betas_ave_te,
        }
        for fname, arr in saves.items():
            path = os.path.join(savedir, fname)
            np.save(path, arr)
            print(f"  Saved {fname}  shape={arr.shape}")

        del betas_roi, betas_roi_ave, betas_tr, betas_te, betas_ave_tr, betas_ave_te

    print(f"\n{'='*60}")
    print(f"Done. All ROIs saved to {savedir}")


if __name__ == "__main__":
    main()
