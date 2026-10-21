"""Single-trial beta estimation for one subject under three first-level timing models.

Models (one GLM per run, 34 trials per run, canonical HRF, identical nuisance model):
  impulse_LSA  each trial an impulse at its onset TR (the original 2021 specification)
  box4_LSA     each trial a 4 s boxcar (the image display duration); all trials in one GLM
  box4_LSS     4 s boxcar, one GLM per trial (trial of interest + one regressor for all other trials)

Usage: python first_level_betas.py sub-sid000013
Requires the raw fMRIPrep outputs and onset files (not distributed; see README).
"""
import io
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nltools.data import Brain_Data, Design_Matrix
from nltools.file_reader import onsets_to_dm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "fMRI" / "derived"
FMRIPREP = ROOT / "fMRI" / "fmriprep" / "fmriprep"
ONSETS = ROOT / "fMRI" / "ONSETS"
PARCELLATION_URL = "http://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz"

TR = 0.6
RUN_LENGTH = 500
N_RUNS = 4
STIM_DURATION = 4.0
HIGH_PASS_SECONDS = 100
MODELS = ("impulse_LSA", "box4_LSA", "box4_LSS")


def load_parcellation():
    path = DERIVED / "parcellation.nii.gz"
    if not path.exists():
        DERIVED.mkdir(parents=True, exist_ok=True)
        Brain_Data(PARCELLATION_URL).to_nifti().to_filename(str(path))
    return nib.load(str(path))


def read_onsets(sub, run):
    raw = (ONSETS / f"IAPS_id{sub}_run{run}_fixed.txt").read_text()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["Stim", "Onset"]).dropna()


def nuisance_matrix(sub, run, bold_2d):
    func = FMRIPREP / sub / "func"
    conf = pd.read_csv(func / f"{sub}_task-viewing_acq-3mm_run-0{run}_bold_confounds.tsv", sep="\t").loc[:, "X":"RotZ"]
    conf = conf - conf.mean()
    base = conf.copy()
    for k in range(6):
        conf[f"rpsq{k + 1}"] = base.iloc[:, k] ** 2
    for k in range(6):
        conf[f"rpdiff{k + 1}"] = base.iloc[:, k].diff()
    for k in range(6):
        conf[f"rpdiffsq{k + 1}"] = base.iloc[:, k].diff() ** 2
    conf["Intercept"] = 1
    trend = np.arange(RUN_LENGTH) - np.mean(np.arange(RUN_LENGTH))
    conf["LinearTrend"] = trend
    conf["QuadraticTrend"] = trend ** 2
    conf = conf.fillna(0)
    frame_diff = np.mean(np.abs(np.diff(bold_2d, axis=0)), axis=1)
    outliers = np.concatenate([
        np.where(frame_diff > frame_diff.mean() + 3 * frame_diff.std())[0],
        np.where(frame_diff < frame_diff.mean() - 3 * frame_diff.std())[0],
    ])
    for i, loc in enumerate(outliers):
        conf[f"FD{i + 1}"] = 0.0
        conf.loc[int(loc), f"FD{i + 1}"] = 1.0
    dm = Design_Matrix(conf, sampling_freq=1 / TR).add_dct_basis(duration=HIGH_PASS_SECONDS)
    return dm.values.astype(np.float64)


def estimate_run(sub, run, parcellation):
    func = FMRIPREP / sub / "func"
    img = nib.load(str(func / f"{sub}_task-viewing_acq-3mm_run-0{run}_bold_space-MNI152NLin2009cAsym_preproc_smoothed.nii.gz"))
    data = img.get_fdata(dtype=np.float32)
    mean_img = data.mean(-1)
    brain = mean_img > np.percentile(mean_img, 60)
    Y = data[brain].T.astype(np.float32)

    onsets = read_onsets(sub, run)
    names = list(onsets.Stim.unique())
    impulse = onsets_to_dm(onsets.copy(), sampling_freq=1 / TR, run_length=RUN_LENGTH, sort=False)
    boxcar_onsets = onsets.copy()
    boxcar_onsets["Duration"] = STIM_DURATION
    boxcar = onsets_to_dm(boxcar_onsets, sampling_freq=1 / TR, run_length=RUN_LENGTH, sort=False)
    assert list(impulse.columns) == names and list(boxcar.columns) == names
    trials = {"impulse_LSA": impulse.convolve().values, "box4_LSA": boxcar.convolve().values}
    nuisance = nuisance_matrix(sub, run, Y)
    n = len(names)

    betas = {}
    for model in ("impulse_LSA", "box4_LSA"):
        weights = np.linalg.pinv(np.hstack([trials[model], nuisance]))[:n]
        betas[model] = weights.astype(np.float32) @ Y
    box = trials["box4_LSA"]
    others = box.sum(1)
    weights = np.empty((n, RUN_LENGTH))
    for k in range(n):
        design = np.hstack([box[:, [k]], (others - box[:, k])[:, None], nuisance])
        weights[k] = np.linalg.pinv(design)[0]
    betas["box4_LSS"] = weights.astype(np.float32) @ Y

    union = parcellation.get_fdata() > 0
    stack = np.zeros(img.shape[:3] + (len(MODELS) * n,), dtype=np.float32)
    for j, model in enumerate(MODELS):
        volume = np.zeros(img.shape[:3] + (n,), dtype=np.float32)
        volume[brain] = betas[model].T
        stack[..., j * n:(j + 1) * n] = volume
    resampled = image.resample_img(
        nib.Nifti1Image(stack, img.affine), target_affine=parcellation.affine,
        target_shape=parcellation.shape, interpolation="continuous",
    ).get_fdata(dtype=np.float32)
    return names, {model: resampled[..., j * n:(j + 1) * n][union] for j, model in enumerate(MODELS)}


def main(sub):
    start = time.time()
    parcellation = load_parcellation()
    images = sorted({s for run in range(1, N_RUNS + 1) for s in read_onsets(sub, run).Stim})
    assert len(images) == 136
    index = {name: i for i, name in enumerate(images)}
    n_union = int((parcellation.get_fdata() > 0).sum())
    out = {m: np.full((136, n_union), np.nan, dtype=np.float32) for m in MODELS}
    for run in range(1, N_RUNS + 1):
        names, per_model = estimate_run(sub, run, parcellation)
        for model in MODELS:
            for k, name in enumerate(names):
                out[model][index[name]] = per_model[model][:, k]
        print(f"{sub} run {run} done ({time.time() - start:.0f}s)", flush=True)
    assert not any(np.isnan(v).any() for v in out.values())
    target = DERIVED / "first_level"
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / f"{sub}.npz", **out, images=np.array(images))


if __name__ == "__main__":
    main(sys.argv[1])
