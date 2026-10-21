"""Run and trial-order structure of the single-trial betas, per timing model.

Per subject and ROI:
  eta^2 (run)        between-run share of each voxel's beta variance over the 136 trials, averaged over voxels
  PC1 share          variance share of the first principal component of the 136 x voxels beta matrix
  |rho| order        |Spearman correlation| of PC1 scores with presentation order
  edge trials        share of the five most extreme PC1 trials that fall in the last two positions of a run
  |PC1| vs norm      correlation of |PC1 score| with each trial's overall pattern magnitude
All are averaged over ROIs and subjects.

Output: Analysis/figs/image_neural_similarity/Study4_run_variance.csv
Usage: python run_variance.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from roi_similarity import MODELS, OUT, load_model, roi_voxel_indices, trial_positions

BASE_MODELS = [m for m in MODELS if not m.endswith("_runeq")]
N_RUNS = 4
N_TRIALS = 136


def main():
    rois = roi_voxel_indices()
    rows = []
    for model in BASE_MODELS:
        subjects, images, data = load_model(model)
        stats = np.zeros((len(subjects), 50, 5))
        for s, sub in enumerate(subjects):
            runs, in_run, order = trial_positions(sub, images)
            for r, idx in enumerate(rois):
                x = data[s][:, idx].astype(np.float64)
                x = x - x.mean(0)
                total = (x ** 2).sum(0)
                between = sum((runs == k).sum() * x[runs == k].mean(0) ** 2 for k in np.unique(runs))
                u, sv, _ = np.linalg.svd(x, full_matrices=False)
                pc1 = u[:, 0] * sv[0]
                extreme = np.argsort(-np.abs(pc1 - np.median(pc1)))[:5]
                stats[s, r] = (
                    np.mean(between / total),
                    sv[0] ** 2 / (sv ** 2).sum(),
                    abs(spearmanr(pc1, order)[0]),
                    np.mean(in_run[extreme] >= 32),
                    abs(np.corrcoef(np.abs(pc1), np.linalg.norm(x, axis=1))[0, 1]),
                )
        mean = stats.mean((0, 1))
        rows.append({
            "Model": model,
            "Run eta^2": round(float(mean[0]), 3),
            "Run eta^2 expected by chance": round((N_RUNS - 1) / (N_TRIALS - 1), 3),
            "PC1 share of variance": round(float(mean[1]), 2),
            "|rho| PC1 vs presentation order": round(float(mean[2]), 2),
            "Extreme PC1 trials at run ends": round(float(mean[3]), 2),
            "Run-end share expected by chance": round(2 / 34, 2),
            "|PC1| vs pattern magnitude": round(float(mean[4]), 2),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "Study4_run_variance.csv", index=False)
    print(frame.T.to_string(header=False))


if __name__ == "__main__":
    main()
