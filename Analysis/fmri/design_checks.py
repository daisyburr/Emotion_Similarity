"""Design and reliability checks behind figures quoted in the Study 3 supplementary text.

1. Image-to-run assignment: how many participant pairs share an assignment, and the mean share of images that a pair
   saw in the same run (0.25 expected under independent random assignment to four runs).
2. Valence labels: agreement between the Study 1 definition (mean Pleasant rating > 30) and the stimulus-set label used
   in Study 3, and the Study 3 two-sample result when the disagreeing images take their Study 1 labels.
3. Split-half reliability (random halves of the images, Spearman-Brown corrected) of the pairwise appraisal similarity and
   of the pairwise neural similarity, per valence.

Inputs: fMRI/derived/first_level/*.npz, fMRI/derived/roi_pairwise_box4_LSA.npz, Online/CIAPS_Appraisal_Recovered_NNMF_SGD.csv
Output: Analysis/figs/image_neural_similarity/Study3_design_checks.csv
Usage: python design_checks.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

import inter_subject_rsa as isrsa
from roi_similarity import DERIVED, OUT, ROOT, trial_positions

SEED = 0
N_PERMUTE = 20_000
N_HALVES = 50


def spearman_brown(r):
    return 2 * r / (1 + r)


def run_assignment_overlap(rows):
    files = sorted((DERIVED / "first_level").glob("sub-sid*.npz"))
    images = list(np.load(files[0])["images"])
    runs = np.stack([next(trial_positions(f.stem, images)) for f in files])
    shared = [(runs[i] == runs[j]).mean() for i in range(len(runs)) for j in range(i + 1, len(runs))]
    rows += [
        ("participants", len(files)),
        ("participant pairs with identical image-to-run assignment", int(sum(np.array(shared) == 1))),
        ("mean share of images in the same run for a pair (chance 0.25)", round(float(np.mean(shared)), 3)),
    ]


def label_agreement(rows):
    ratings = pd.read_csv(ROOT / "Online" / "CIAPS_Appraisal_Recovered_NNMF_SGD.csv", usecols=["Image", "Rating", "Condition"])
    pleasant = ratings[ratings.Condition == "Pleasant"].groupby("Image").Rating.mean()
    study1 = (pleasant > 30).rename("study1_pos")
    info = pd.read_csv(OUT / "image_valence_category.csv", index_col="Image")
    joined = pd.concat([study1, info.Valence == "pos"], axis=1, join="inner")
    joined.columns = ["study1_pos", "study3_pos"]
    disagree = joined[joined.study1_pos != joined.study3_pos]
    rows += [("images in both label sets", len(joined)), ("images whose valence label differs", len(disagree)),
             ("images whose valence label differs (names)", "; ".join(disagree.index))]

    z = np.arctanh(pd.read_csv(OUT / "ROI_Image_Similarity_box4_LSA.csv", index_col=0).reindex(info.index).values)
    rng = np.random.default_rng(SEED)
    for name, label in (("Study 3 labels", (info.Valence == "pos").values), ("Study 1 labels", joined.study1_pos.reindex(info.index).values)):
        observed = z[label].mean(0) - z[~label].mean(0)
        count = np.zeros(50)
        for _ in range(N_PERMUTE):
            perm = rng.permutation(label)
            count += np.abs(z[perm].mean(0) - z[~perm].mean(0)) >= np.abs(observed)
        reject = multipletests((count + 1) / (N_PERMUTE + 1), alpha=0.05, method="fdr_bh")[0]
        rows.append((f"primary model, FDR-significant ROIs with {name} (negative > positive / positive > negative)",
                     f"{int(reject.sum())} ({int((reject & (observed < 0)).sum())} / {int((reject & (observed > 0)).sum())})"))


def split_half_reliability(rows):
    data = np.load(DERIVED / "roi_pairwise_box4_LSA.npz")
    subjects = [str(s) for s in data["subjects"]]
    images = [str(i) for i in data["images"]]
    upper = np.triu_indices(len(subjects), 1)
    info = pd.read_csv(OUT / "image_valence_category.csv", index_col="Image")
    valence = info.loc[[i.split("_", 1)[1] for i in images], "Valence"].values
    neural = isrsa.fisher(data["corr"][:, :, upper[0], upper[1]].astype(np.float64))
    behavioral = isrsa.behavioral_pairs(subjects, images, upper)
    rng = np.random.default_rng(SEED)
    for label in ("pos", "neg"):
        index = np.where(valence == label)[0]
        r_behavior, r_neural = [], []
        for _ in range(N_HALVES):
            perm = rng.permutation(index)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
            r_behavior.append(spearmanr(behavioral[a].mean(0), behavioral[b].mean(0))[0])
            r_neural.append(np.mean([spearmanr(neural[r][a].mean(0), neural[r][b].mean(0))[0] for r in range(50)]))
        rows += [(f"split-half reliability of pairwise appraisal similarity, {label} images", round(float(spearman_brown(np.mean(r_behavior))), 2)),
                 (f"split-half reliability of pairwise neural similarity (mean over ROIs), {label} images", round(float(spearman_brown(np.mean(r_neural))), 2))]


if __name__ == "__main__":
    rows = []
    for step in (run_assignment_overlap, label_agreement, split_half_reliability):
        step(rows)
    frame = pd.DataFrame(rows, columns=["Check", "Value"])
    frame.to_csv(OUT / "Study3_design_checks.csv", index=False)
    print(frame.to_string(index=False))
