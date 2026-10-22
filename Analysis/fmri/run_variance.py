"""Run structure of the single-trial betas, per timing model.

Per subject and ROI: eta^2 (run), the between-run share of each voxel's beta variance over the 136 trials, averaged over voxels.
Averaged over ROIs and subjects, and compared with the share expected if the 136 images were assigned to the four runs at random.

Output: Analysis/figs/image_neural_similarity/Study3_run_variance.csv
Usage: python run_variance.py
"""
import numpy as np
import pandas as pd

from roi_similarity import MODELS, OUT, load_model, roi_voxel_indices, trial_positions

BASE_MODELS = [m for m in MODELS if not m.endswith("_runeq")]
N_RUNS = 4
N_TRIALS = 136


def main():
    rois = roi_voxel_indices()
    rows = []
    for model in BASE_MODELS:
        subjects, images, data = load_model(model)
        eta = np.zeros((len(subjects), 50))
        for s, sub in enumerate(subjects):
            runs, _, _ = trial_positions(sub, images)
            for r, idx in enumerate(rois):
                x = data[s][:, idx].astype(np.float64)
                x = x - x.mean(0)
                total = (x ** 2).sum(0)
                between = sum((runs == k).sum() * x[runs == k].mean(0) ** 2 for k in np.unique(runs))
                eta[s, r] = np.mean(between / total)
        rows.append({
            "Model": model,
            "Run eta^2": round(float(eta.mean()), 3),
            "Run eta^2 expected by chance": round((N_RUNS - 1) / (N_TRIALS - 1), 3),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "Study3_run_variance.csv", index=False)
    print(frame.T.to_string(header=False))


if __name__ == "__main__":
    main()
