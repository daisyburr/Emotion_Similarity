"""Supplementary Study 4 analyses: category-level inference and evoked-amplitude controls.

1. Category-level inference. Images sit in valence-pure stimulus categories (23 category x valence clusters), so the
   image-level permutation test treats non-independent images as independent. Here the cluster labels themselves are
   permuted: every one of the C(23, 11) = 1,352,078 ways of assigning 11 clusters to the positive set is enumerated
   (exact p, no Monte Carlo error). Statistic: image-weighted mean difference in Fisher-z ISS (positive - negative).
   Variants: odd single-image cluster dropped, and a t-test on cluster means.
2. Evoked amplitude. ISS rises with how strongly an image evokes the picture-response pattern shared across subjects.
   Subjects are split at random into halves (100 splits): image amplitude (projection onto the template built from half A)
   is measured independently of ISS (computed in half B). Reported: the valence effect on ISS, on ISS adjusted for
   amplitude (linear + quadratic), on amplitude itself, and on item-specific ISS (the template direction projected out).

Inputs: Analysis/figs/image_neural_similarity/ROI_Image_Similarity_<model>.csv, image_valence_category.csv,
        fMRI/derived/roi_pairwise_<model>.npz (from roi_similarity.py)
Outputs: Study4_category_level_by_model.csv, Study4_category_level_by_ROI.csv, Study4_amplitude_summary.csv
Usage: python supplement_analyses.py
"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from roi_similarity import DERIVED, OUT

MODELS = ["published_2021", "impulse_LSA", "box4_LSA", "box4_LSS", "impulse_LSA_runeq", "box4_LSA_runeq", "box4_LSS_runeq"]
AMPLITUDE_MODELS = ["impulse_LSA", "box4_LSA", "box4_LSS"]
N_SPLITS = 100
SEED = 1


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


def tvalue(y, regressors):
    X = np.column_stack([np.ones(len(y))] + regressors)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = np.sqrt((resid ** 2).sum() / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X).diagonal())
    return (beta / se)[1], len(y) - X.shape[1]


def amplitude_analysis(model, valence, rng):
    data = np.load(DERIVED / f"roi_pairwise_{model}.npz")
    corr, norm, proj, template = data["corr"], data["norm"], data["proj"], data["template"]
    n_sub = corr.shape[2]
    half = n_sub // 2
    splits = [rng.permutation(n_sub) for _ in range(N_SPLITS)]
    outcomes = ("raw", "adjusted", "amplitude", "item_specific")
    t = {k: np.zeros((N_SPLITS, 50)) for k in outcomes}
    dfs = {}
    amp_iss = np.zeros((N_SPLITS, 50))
    for r in range(50):
        for k, perm in enumerate(splits):
            A, B = perm[:half], perm[half:]
            iu = np.triu_indices(len(B), 1)
            a, b = B[iu[0]], B[iu[1]]
            t2 = template[r][np.ix_(A, A)].mean()
            P = proj[r][:, :, A].mean(-1)
            amplitude = (P[A] / t2).mean(0)
            raw = np.arctanh(np.clip(corr[r][:, a, b].mean(1), -0.999, 0.999))
            p_img = (P / np.sqrt(t2)).T[:, B]
            gram = corr[r][:, a, b] * norm[r][:, a] * norm[r][:, b]
            numerator = gram - p_img[:, iu[0]] * p_img[:, iu[1]]
            denominator = np.sqrt(
                np.clip(norm[r][:, a] ** 2 - p_img[:, iu[0]] ** 2, 1e-12, None)
                * np.clip(norm[r][:, b] ** 2 - p_img[:, iu[1]] ** 2, 1e-12, None)
            )
            item = np.arctanh(np.clip((numerator / denominator).mean(1), -0.999, 0.999))
            t["raw"][k, r], dfs["raw"] = tvalue(raw, [valence])
            t["adjusted"][k, r], dfs["adjusted"] = tvalue(raw, [valence, amplitude, amplitude ** 2])
            t["amplitude"][k, r], dfs["amplitude"] = tvalue(amplitude, [valence])
            t["item_specific"][k, r], dfs["item_specific"] = tvalue(item, [valence])
            amp_iss[k, r] = np.corrcoef(amplitude, raw)[0, 1]
    rows = []
    labels = {
        "raw": "valence effect on ISS (half B)",
        "adjusted": "... adjusted for independent amplitude (half A)",
        "amplitude": "valence effect on independent amplitude",
        "item_specific": "valence effect on item-specific ISS (template removed)",
    }
    for key in outcomes:
        tt = t[key]
        reliable = (np.abs(tt) > stats.t.isf(0.025, dfs[key])).mean(0) >= 0.5
        rows.append({
            "Model": model, "Outcome": labels[key],
            "Mean t across ROIs": round(float(tt.mean()), 2),
            "ROIs with mean t < -2": int((tt.mean(0) < -2).sum()),
            "ROIs significant in >= 50% of splits": int(reliable.sum()),
        })
    rows.append({"Model": model, "Outcome": "correlation of independent amplitude with ISS across images (mean over ROIs)",
                 "Mean t across ROIs": round(float(amp_iss.mean()), 2)})
    return rows


def summarize_amplitude():
    info, _ = load_clusters()
    names = [i for i in pd.read_csv(OUT / "ROI_Image_Similarity_box4_LSA.csv", index_col=0).index]
    assert names == list(info.index)
    valence = (info["Valence"] == "pos").astype(float).values
    rng = np.random.default_rng(SEED)
    rows = []
    for model in AMPLITUDE_MODELS:
        data_images = [str(i).split("_", 1)[1] for i in np.load(DERIVED / f"roi_pairwise_{model}.npz")["images"]]
        rows += amplitude_analysis(model, np.array([valence[names.index(n)] for n in data_images]), rng)
        print(model, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "Study4_amplitude_summary.csv", index=False)
    return frame


if __name__ == "__main__":
    print(summarize_category_level().to_string(index=False))
    print(summarize_amplitude().to_string(index=False))
