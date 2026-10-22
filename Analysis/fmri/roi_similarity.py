"""ROI inter-subject similarity (ISS) from first-level betas.

For every timing model this writes
  Analysis/figs/image_neural_similarity/ROI_Image_Similarity_<model>.csv   136 images x 50 ROIs (mean pairwise r)
  fMRI/derived/roi_pairwise_<model>.npz   compact subject-level arrays that reproduce every downstream analysis
                                          without voxel data (pairwise pattern correlations per ROI)

Models ending in _runeq equalize the run means of each subject's betas before correlating (ISS csv only).

Usage: python roi_similarity.py [model ...]
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "fMRI" / "derived"
OUT = ROOT / "Analysis" / "figs" / "image_neural_similarity"
ONSETS = ROOT / "fMRI" / "ONSETS"
N_RUNS = 4
MODELS = ("impulse_LSA", "box4_LSA", "box4_LSS", "impulse_LSA_runeq", "box4_LSA_runeq", "box4_LSS_runeq")


def roi_voxel_indices():
    parcellation = nib.load(str(DERIVED / "parcellation.nii.gz")).get_fdata()
    labels = np.round(parcellation[parcellation > 0]).astype(int)
    return [np.where(labels == r)[0] for r in range(1, 51)]


def trial_positions(sub, images):
    """Run number, position within run (0-33) and position in the whole session for each image."""
    run_of, in_run, overall = {}, {}, {}
    for run in range(1, N_RUNS + 1):
        raw = (ONSETS / f"IAPS_id{sub}_run{run}_fixed.txt").read_text().replace("\r\n", "\n").replace("\r", "\n")
        for k, line in enumerate(l for l in raw.split("\n") if l.strip()):
            name = line.split("\t")[0]
            run_of[name], in_run[name], overall[name] = run, k, len(overall)
    return (np.array([d[i] for i in images]) for d in (run_of, in_run, overall))


def run_of_image(sub, images):
    return next(trial_positions(sub, images))


def equalize_runs(betas, runs):
    """Give every run the same mean pattern (the subject's grand mean), removing run offsets only."""
    out = betas.copy()
    grand = betas.mean(0)
    for run in np.unique(runs):
        sel = runs == run
        out[sel] = betas[sel] - betas[sel].mean(0) + grand
    return out


def load_model(model):
    files = sorted((DERIVED / "first_level").glob("sub-sid*.npz"))
    arrays = [np.load(f) for f in files]
    images = list(arrays[0]["images"])
    assert all(list(a["images"]) == images for a in arrays)
    base = model.removesuffix("_runeq")
    data = np.stack([a[base] for a in arrays]).astype(np.float32)
    if model.endswith("_runeq"):
        data = np.stack([equalize_runs(d, run_of_image(f.stem, images)) for d, f in zip(data, files)])
    return [f.stem for f in files], images, data


def summarize(model):
    subjects, images, data = load_model(model)
    n_sub, n_img = data.shape[:2]
    upper = np.triu_indices(n_sub, 1)
    iss = np.empty((n_img, 50))
    corr = np.empty((50, n_img, n_sub, n_sub), dtype=np.float32)
    for r, idx in enumerate(roi_voxel_indices()):
        x = data[:, :, idx].astype(np.float64)
        x -= x.mean(-1, keepdims=True)
        gram = np.einsum("siv,tiv->ist", x, x, optimize=True)
        length = np.sqrt(np.einsum("iss->is", gram))
        corr[r] = gram / (length[:, :, None] * length[:, None, :])
        iss[:, r] = corr[r][:, upper[0], upper[1]].mean(1)
    names = [i.split("_", 1)[1] for i in images]
    frame = pd.DataFrame(iss, index=names, columns=[f"ROI_{r}" for r in range(50)]).sort_index()
    frame.to_csv(OUT / f"ROI_Image_Similarity_{model}.csv")
    pd.DataFrame({
        "Image": names,
        "Valence": ["pos" if i.startswith("pos_") else "neg" for i in images],
        "Category": [n.split("-")[0] for n in names],
    }).sort_values("Image").to_csv(OUT / "image_valence_category.csv", index=False)
    if not model.endswith("_runeq"):
        np.savez_compressed(
            DERIVED / f"roi_pairwise_{model}.npz", corr=corr,
            subjects=np.array(subjects), images=np.array(images),
        )
    print(f"{model}: ISS {frame.shape}, mean r = {iss.mean():.4f}", flush=True)


if __name__ == "__main__":
    for model in (sys.argv[1:] or MODELS):
        summarize(model)
