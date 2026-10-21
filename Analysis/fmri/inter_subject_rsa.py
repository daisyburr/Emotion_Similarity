"""Inter-subject RSA: does neural similarity between subject pairs track appraisal similarity between the same pairs?

Per ROI and per valence (positive / negative images):
  neural similarity   mean over images of Fisher-z pairwise pattern correlation between two subjects
  behavioral similarity   mean over images of Fisher-z correlation of the two subjects' 12-dimensional appraisal profiles
  statistic           Spearman correlation across the 465 subject pairs; Mantel permutation test (subject labels), one-tailed
Positive-vs-negative difference in coupling: permutation of image labels, and of stimulus-category labels.
Variant: subject main effects (row and column means) removed from both matrices before correlating.

Inputs: fMRI/derived/roi_pairwise_<model>.npz (from roi_similarity.py), Analysis/IAPS_rating.csv
Output: Analysis/figs/image_neural_similarity/Study4_IS_RSA_by_ROI.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Analysis" / "figs" / "image_neural_similarity"
MODEL = "box4_LSA"
N_MANTEL = 10_000
N_CONTRAST = 5_000
SEED = 0
APPRAISALS = [
    "Narrative_Rating", "Immoral_Rating", "Threatened_Rating", "Avoid_Rating", "SociallyAcceptable_Rating",
    "Interact_Rating", "Contamination_Rating", "Suffering_Rating", "Affection_Rating", "Desire_Rating",
    "Astonished_Rating", "Relate_Rating",
]


def fisher(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def behavioral_pairs(subjects, images, upper):
    ratings = pd.read_csv(ROOT / "Analysis" / "IAPS_rating.csv", index_col=0)
    ratings = ratings[ratings.Subject.isin(subjects)]
    names = [i.split("_", 1)[1] for i in images]
    out = np.full((len(images), len(upper[0])), np.nan)
    for i, name in enumerate(names):
        profile = ratings[ratings.Image == name].set_index("Subject").loc[subjects, APPRAISALS].values
        out[i] = fisher(np.corrcoef(profile))[upper]
    return out


def remove_subject_effects(vec, n_sub, upper):
    m = np.full((n_sub, n_sub), np.nan)
    m[upper] = vec
    m[(upper[1], upper[0])] = vec
    row = np.nanmean(m, axis=1)
    resid = m - row[:, None] - row[None, :] + np.nanmean(m)
    return resid[upper]


def spearman_rows(x, y):
    xr = np.apply_along_axis(rankdata, -1, x)
    yr = np.apply_along_axis(rankdata, -1, y)
    xr = xr - xr.mean(-1, keepdims=True)
    yr = yr - yr.mean(-1, keepdims=True)
    return (xr * yr).sum(-1) / np.sqrt((xr ** 2).sum(-1) * (yr ** 2).sum(-1))


def mantel(neural, behavior, upper, n_sub, rng):
    ranks_b = rankdata(behavior)
    pair_index = np.zeros((n_sub, n_sub), dtype=int)
    pair_index[upper] = np.arange(len(upper[0]))
    pair_index[(upper[1], upper[0])] = np.arange(len(upper[0]))
    ranks_n = np.apply_along_axis(rankdata, -1, neural)
    ranks_n = ranks_n - ranks_n.mean(-1, keepdims=True)
    ranks_n /= np.linalg.norm(ranks_n, axis=-1, keepdims=True)
    observed = spearman_rows(neural, np.broadcast_to(behavior, neural.shape))
    null = np.empty((N_MANTEL, neural.shape[0]))
    for start in range(0, N_MANTEL, 1000):
        perm = np.stack([rng.permutation(n_sub) for _ in range(1000)])
        idx = pair_index[perm[:, upper[0]], perm[:, upper[1]]]
        b = ranks_b[idx]
        b = b - b.mean(-1, keepdims=True)
        b /= np.linalg.norm(b, axis=-1, keepdims=True)
        null[start:start + 1000] = b @ ranks_n.T
    p = (np.sum(null >= observed, axis=0) + 1) / (N_MANTEL + 1)
    return observed, p


def set_means(weights, zn, zb):
    neural = np.einsum("pi,rik->prk", weights, zn, optimize=True)
    behavior = weights @ zb
    return neural, behavior


def coupling_difference(zn, zb, valence, clusters, rng, mode):
    n_img = len(valence)
    is_pos = valence == "pos"
    cluster_ids = np.unique(clusters)
    cluster_pos = np.array([is_pos[clusters == c].all() for c in cluster_ids])
    null = np.empty((N_CONTRAST, zn.shape[0]))
    for start in range(0, N_CONTRAST, 250):
        size = min(250, N_CONTRAST - start)
        w_pos = np.zeros((size, n_img))
        w_neg = np.zeros((size, n_img))
        for p in range(size):
            if mode == "image":
                label = rng.permutation(is_pos)
            else:
                assigned = rng.permutation(cluster_pos)
                label = np.isin(clusters, cluster_ids[assigned])
            w_pos[p, label] = 1.0 / label.sum()
            w_neg[p, ~label] = 1.0 / (~label).sum()
        n_p, b_p = set_means(w_pos, zn, zb)
        n_n, b_n = set_means(w_neg, zn, zb)
        r_p = np.stack([spearman_rows(n_p[k], np.broadcast_to(b_p[k], n_p[k].shape)) for k in range(size)])
        r_n = np.stack([spearman_rows(n_n[k], np.broadcast_to(b_n[k], n_n[k].shape)) for k in range(size)])
        null[start:start + size] = r_p - r_n
    return null


def main():
    rng = np.random.default_rng(SEED)
    data = np.load(ROOT / "fMRI" / "derived" / f"roi_pairwise_{MODEL}.npz")
    subjects = [str(s) for s in data["subjects"]]
    images = [str(i) for i in data["images"]]
    n_sub = len(subjects)
    upper = np.triu_indices(n_sub, 1)
    info = pd.read_csv(OUT / "image_valence_category.csv", index_col="Image")
    names = [i.split("_", 1)[1] for i in images]
    valence = info.loc[names, "Valence"].values
    clusters = np.array([f"{info.loc[n, 'Category']}{'+' if info.loc[n, 'Valence'] == 'pos' else '-'}" for n in names])

    zn = fisher(data["corr"][:, :, upper[0], upper[1]].astype(np.float64))
    zb = behavioral_pairs(subjects, images, upper)
    assert not np.isnan(zb).any()
    result = pd.DataFrame({"ROI": range(50)})

    for tag, centered in (("", False), ("sc_", True)):
        zn_v, zb_v = zn, zb
        if centered:
            zn_v = np.stack([[remove_subject_effects(zn[r, i], n_sub, upper) for i in range(zn.shape[1])] for r in range(50)])
            zb_v = np.stack([remove_subject_effects(zb[i], n_sub, upper) for i in range(zb.shape[0])])
        stats = {}
        for label in ("pos", "neg"):
            sel = valence == label
            neural = zn_v[:, sel].mean(1)
            behavior = zb_v[sel].mean(0)
            r, p = mantel(neural, behavior, upper, n_sub, rng)
            result[f"{tag}r_{label}"] = r
            result[f"{tag}p_{label}"] = p
            result[f"{tag}q_{label}"] = multipletests(p, alpha=0.05, method="fdr_bh")[1]
            stats[label] = r
        delta = stats["pos"] - stats["neg"]
        result[f"{tag}delta_pos_minus_neg"] = delta
        for mode in ("image", "category"):
            null = coupling_difference(zn_v, zb_v, valence, clusters, rng, mode)
            p = (np.sum(np.abs(null) >= np.abs(delta), axis=0) + 1) / (N_CONTRAST + 1)
            result[f"{tag}p_delta_{mode}"] = p
            result[f"{tag}q_delta_{mode}"] = multipletests(p, alpha=0.05, method="fdr_bh")[1]
    result.to_csv(OUT / "Study4_IS_RSA_by_ROI.csv", index=False)
    print(result.describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]
    main()
