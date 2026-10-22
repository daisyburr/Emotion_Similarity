"""Supplementary Study 4 analysis: category-level inference.

1. Category-level inference. Images sit in valence-pure stimulus categories (23 category x valence clusters), so the
   image-level permutation test treats non-independent images as independent. Here the cluster labels themselves are
   permuted: every one of the C(23, 11) = 1,352,078 ways of assigning 11 clusters to the positive set is enumerated
   (exact p, no Monte Carlo error). Statistic: image-weighted mean difference in Fisher-z ISS (positive - negative).
   Variants: odd single-image cluster dropped, and a t-test on cluster means.

Inputs: Analysis/figs/image_neural_similarity/ROI_Image_Similarity_<model>.csv, image_valence_category.csv,
        fMRI/derived/roi_pairwise_<model>.npz (from roi_similarity.py)
Outputs: Study4_category_level_by_model.csv, Study4_category_level_by_ROI.csv
Usage: python supplement_analyses.py
"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from roi_similarity import OUT

MODELS = ["published_2021", "impulse_LSA", "box4_LSA", "box4_LSS", "impulse_LSA_runeq", "box4_LSA_runeq", "box4_LSS_runeq"]


def load_clusters():
    info = pd.read_csv(OUT / "image_valence_category.csv", index_col="Image")
    cluster = info["Category"] + info["Valence"].map({"pos": "+", "neg": "-"})
    return info, cluster


def enumerate_subsets(n, k, chunk=200_000):
    combos = combinations(range(n), k)
    while True:
        block = np.array(list(_take(combos, chunk)))
        if len(block) == 0:
            return
        indicator = np.zeros((len(block), n), dtype=np.float32)
        np.put_along_axis(indicator, block, 1.0, axis=1)
        yield indicator


def _take(iterator, n):
    for _ in range(n):
        try:
            yield next(iterator)
        except StopIteration:
            return


def category_level(z, info, cluster, drop_singletons=False):
    """Exact cluster-label permutation of the positive-minus-negative Fisher-z ISS difference, per ROI."""
    keep = np.ones(len(z), dtype=bool)
    if drop_singletons:
        keep = cluster.map(cluster.value_counts()).values > 1
    z, cluster = z[keep], cluster[keep]
    ids = sorted(cluster.unique())
    index = {c: i for i, c in enumerate(ids)}
    member = np.zeros((len(cluster), len(ids)))
    member[np.arange(len(cluster)), cluster.map(index).values] = 1
    size = member.sum(0)
    sums = member.T @ z
    is_pos = np.array([c.endswith("+") for c in ids])
    n_pos = int(is_pos.sum())
    total_n, total_sum = size.sum(), sums.sum(0)

    def difference(select):
        n1 = select @ size
        s1 = select @ sums
        return s1 / n1[:, None] - (total_sum - s1) / (total_n - n1)[:, None]

    observed = difference(is_pos[None].astype(np.float64))[0]
    exceed = np.zeros(z.shape[1])
    n_total = 0
    for indicator in enumerate_subsets(len(ids), n_pos):
        null = difference(indicator.astype(np.float64))
        exceed += (np.abs(null) >= np.abs(observed) - 1e-12).sum(0)
        n_total += len(indicator)
    assert n_total == comb(len(ids), n_pos)
    p = exceed / n_total
    t_p = stats.ttest_ind(sums[is_pos] / size[is_pos, None], sums[~is_pos] / size[~is_pos, None], axis=0).pvalue
    return observed, p, t_p, len(ids), n_total


def summarize_category_level():
    info, cluster = load_clusters()
    by_model, by_roi = [], {}
    for model in MODELS:
        path = OUT / "original_impulse_2021" / "ROI_Image_Similarity.csv" if model == "published_2021" else OUT / f"ROI_Image_Similarity_{model}.csv"
        z = np.arctanh(pd.read_csv(path, index_col=0).reindex(info.index).values)
        for label, drop in (("all clusters", False), ("odd single-image cluster dropped", True)):
            diff, p, t_p, n_clusters, n_perm = category_level(z, info, cluster, drop)
            rej = multipletests(p, alpha=0.05, method="fdr_bh")[0]
            rej_t = multipletests(t_p, alpha=0.05, method="fdr_bh")[0]
            by_model.append({
                "Model": model, "Clusters": label, "n clusters": n_clusters, "Relabelings": n_perm,
                "ROIs neg > pos (of 50)": int((diff < 0).sum()),
                "ROIs p < .05 uncorrected (exact permutation)": int((p < 0.05).sum()),
                "FDR-significant ROIs (exact permutation)": int(rej.sum()),
                "Significant, neg > pos": int((rej & (diff < 0)).sum()),
                "Significant, pos > neg": int((rej & (diff > 0)).sum()),
                "FDR-significant ROIs (t-test on cluster means)": int(rej_t.sum()),
            })
            if model == "box4_LSA" and not drop:
                by_roi = {"ROI": range(50), "diff_pos_minus_neg": diff, "p_exact": p,
                          "q_exact": multipletests(p, alpha=0.05, method="fdr_bh")[1]}
        print(model, flush=True)
    pd.DataFrame(by_model).to_csv(OUT / "Study4_category_level_by_model.csv", index=False)
    pd.DataFrame(by_roi).to_csv(OUT / "Study4_category_level_by_ROI.csv", index=False)
    return pd.DataFrame(by_model)


if __name__ == "__main__":
    print(summarize_category_level().to_string(index=False))
