"""Robustness checks for Studies 1-3 (supplementary; every figure quoted in the review of these studies comes from here).

A. Valence definitions: agreement between the Study 1 (mean Pleasant > 30), Study 3 (category map) and Study 4 (stimulus-set) labels.
B. Study 1 ratings: the recovered file holds model predictions for every cell; its range, and the valence difference in rater variance
   in the recovered vs the raw ratings.
C. Study 1 on raw ratings (needs the psiTurk dump of an earlier, independent sample; set RAW_DUMP; skipped if absent). Each rater rated
   one dimension per image, so only per-dimension quantities are available: rater disagreement, disagreement after removing the
   dependence of spread on mean level, split-half reliability of the 19-dimension mean profile, and the dimensionality of the
   group-mean appraisal matrix (participant bootstrap). Dimension codes in this dump do not match the notebook's, so all 19 are used.
D. Study 2: similarity with and without arousal in the feature set; valence effect adjusted for arousal, categories endorsed per rater
   and rater count.
E. Study 3: leave-one-subject-out cross-validated R2 under the notebook's 13-component PLS (equivalent to OLS), regularized variants,
   and a valence-blind nested cross-validation that chooses model complexity inside the training raters.
F. Study 3 following Cowen & Keltner (2017, PNAS): leave-one-out cross-validated linear and k-nearest-neighbor regression of the
   emotion ratings on the appraisal ratings across images (rater means), run separately within positive and negative images;
   explained variance divided by the explainable variance (split-half reliability of each emotion rating across images, Spearman-Brown
   corrected); k chosen from 1-50 as the value with the greatest average explained variance; the positive-minus-negative difference
   is bootstrapped by resampling stimulus categories within each valence.
Category-level p-values use the exact cluster-label permutation in fmri/supplement_analyses.py.

Usage: python robustness_studies1to3.py        (RAW_DUMP=/path/to/dump.tsv to include part C)
Output: Analysis/figs/robustness_studies1to3.csv
"""
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "fmri"))
import supplement_analyses as S  # noqa: E402
from study3_transformation import (CATEGORY_VALENCE, FEATURES, TARGETS, cowen_keltner, image_table,  # noqa: E402
                                   nested_prediction)

RAW_DUMP = os.environ.get("RAW_DUMP", "")
SEED = 0
rows = []


def add(section, check, value):
    rows.append((section, check, value))
    print(f"[{section}] {check}: {value}", flush=True)


category_info = image_table


def compare(section, label, x, info, cluster, lower_is_positive_effect=False):
    pos = (info.Valence == "pos").values
    a, b = x[pos], x[~pos]
    t, p = stats.ttest_ind(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    _, pe, _, n_clusters, n_relabel = S.category_level(x[:, None], info, cluster)
    add(section, label, f"positive {a.mean():.3f}, negative {b.mean():.3f}, d = {d:+.2f}, image-level p = {p:.2g}, "
                        f"category-level exact p = {pe[0]:.2g} ({n_clusters} clusters)")


def valence_labels():
    online = pd.read_csv(ROOT / "Online" / "CIAPS_Appraisal_Recovered_NNMF_SGD.csv", usecols=["Image", "Rating", "Condition"])
    pleasant = online[online.Condition == "Pleasant"].groupby("Image").Rating.mean()
    study1 = (pleasant > 30).map({True: "pos", False: "neg"})
    study4 = pd.read_csv(S.OUT / "image_valence_category.csv", index_col="Image").Valence
    study3 = pd.Series({i: CATEGORY_VALENCE[i[:2]] for i in study1.index})
    frame = pd.concat([study1.rename("study1"), study3.rename("study3"), study4.rename("study4")], axis=1)
    for a, b in (("study1", "study3"), ("study1", "study4"), ("study3", "study4")):
        diff = frame[frame[a] != frame[b]]
        add("A", f"images labelled differently, {a} vs {b}", f"{len(diff)} ({'; '.join(diff.index)})")


def recovered_ratings():
    online = pd.read_csv(ROOT / "Online" / "CIAPS_Appraisal_Recovered_NNMF_SGD.csv")
    r = online.Rating.values
    add("B", "recovered ratings: range", f"{r.min():.1f} to {r.max():.1f}; share outside 0-100: {((r < 0) | (r > 100)).mean():.3f}")
    wide = online.pivot_table(index=["Subject", "Image"], columns="Condition", values="Rating")
    images = sorted(online.Image.unique())
    info, _ = category_info(images)
    var = wide.groupby(level="Image").var(ddof=1).mean(1).reindex(images)
    pos = (info.Valence == "pos").values
    add("B", "recovered ratings: mean across-participant variance, positive / negative / ratio (19 dimensions)",
        f"{var[pos].mean():.0f} / {var[~pos].mean():.0f} / {var[pos].mean() / var[~pos].mean():.2f}")


def parse_raw(path):
    table = pd.read_csv(path, sep="\t", low_memory=False)
    out = []
    for uid, text in zip(table.uniqueid, table.datastring):
        if isinstance(text, str):
            for m in re.finditer(r'"image":"([^"]+)","dimension":(\d+),"response":(-?\d+)', text.replace('\\"', '"')):
                out.append((uid, m.group(1), int(m.group(2)), int(m.group(3))))
    return pd.DataFrame(out, columns=["Subject", "Image", "Dim", "Rating"])


def raw_ratings():
    if not RAW_DUMP or not Path(RAW_DUMP).exists():
        add("C", "raw-rating analyses", "skipped (RAW_DUMP not set)")
        return
    raw = parse_raw(RAW_DUMP)
    images = sorted(raw.Image.unique())
    info, cluster = category_info(images)
    pos = (info.Valence == "pos").values
    dims = sorted(raw.Dim.unique())
    add("C", "raw sample", f"{raw.Subject.nunique()} raters, {len(raw)} ratings, {raw.groupby(['Image', 'Dim']).size().mean():.1f} raters per image-dimension")
    g = raw.groupby(["Image", "Dim"]).Rating
    mean = g.mean().unstack().reindex(images)[dims]
    spread = g.std().unstack().reindex(images)[dims]
    compare("C", "rater SD, mean over 19 dimensions (lower = more agreement)", spread.mean(1).values, info, cluster)
    residual = np.zeros(spread.shape)
    for j in range(spread.shape[1]):
        X = np.column_stack([np.ones(len(images)), mean.values[:, j], mean.values[:, j] ** 2])
        residual[:, j] = spread.values[:, j] - X @ np.linalg.lstsq(X, spread.values[:, j], rcond=None)[0]
    compare("C", "rater SD after removing the mean-level trend within each dimension", residual.mean(1), info, cluster)

    rng = np.random.default_rng(SEED)
    cells = {key: v.values for key, v in raw.groupby(["Image", "Dim"]).Rating}
    reliability = np.zeros((50, len(images)))
    for s in range(50):
        for i, image in enumerate(images):
            half_a, half_b = [], []
            for dim in dims:
                v = cells[(image, dim)]
                perm = rng.permutation(len(v))
                h = len(v) // 2
                half_a.append(v[perm[:h]].mean())
                half_b.append(v[perm[h:2 * h]].mean())
            reliability[s, i] = np.corrcoef(half_a, half_b)[0, 1]
    r = np.tanh(np.arctanh(np.clip(reliability, -0.999, 0.999)).mean(0))
    compare("C", "split-half reliability of the 19-dimension mean profile (Spearman-Brown)", 2 * r / (1 + r), info, cluster)

    def components(matrix):
        matrix = matrix - matrix.mean(0)
        variance = np.linalg.svd(matrix, compute_uv=False) ** 2
        return int(np.searchsorted(np.cumsum(variance) / variance.sum(), 0.9) + 1)
    add("C", "components for 90% variance of the group-mean appraisal matrix, positive / negative",
        f"{components(mean.values[pos])} / {components(mean.values[~pos])}")
    subjects = raw.Subject.unique()
    index = raw.Subject.map({s: k for k, s in enumerate(subjects)}).values
    differences = []
    for _ in range(200):
        weights = np.bincount(rng.choice(len(subjects), len(subjects)), minlength=len(subjects))[index]
        weighted = (raw.Rating * weights).groupby([raw.Image, raw.Dim]).sum() / pd.Series(weights, index=raw.index).groupby([raw.Image, raw.Dim]).sum()
        m = weighted.unstack().reindex(images)[dims].values
        differences.append(components(m[pos]) - components(m[~pos]))
    add("C", "participant bootstrap, positive minus negative components (mean, 95% interval)",
        f"{np.mean(differences):+.2f} {np.percentile(differences, [2.5, 97.5]).tolist()}")


def isc(wide, id_col, columns):
    out = {}
    for stim, group in wide.groupby(id_col):
        x = group[columns].astype(float).values
        if len(x) < 2:
            continue
        with np.errstate(all="ignore"):
            z = np.arctanh(np.corrcoef(x)[np.triu_indices(len(x), 1)])
        z = z[np.isfinite(z)]
        if len(z):
            out[stim] = np.tanh(z.mean())
    return pd.Series(out)


def study2():
    ck = HERE / "CowenKeltner"
    if not (ck / "indiv_resp_dimensions.csv").exists():
        add("D", "Study 2 checks", "skipped (Cowen & Keltner files not present)")
        return
    dims = pd.read_csv(ck / "indiv_resp_dimensions.csv")
    cats = pd.read_csv(ck / "indiv_resp_categories.csv")
    avg = pd.read_csv(ck / "averages_dimensions.csv").set_index("stimulus_id")
    add("D", "dimensions rated", f"{dims.dim_name.nunique()}: {', '.join(sorted(dims.dim_name.unique()))}")
    wide = dims.pivot_table(index=["stimulus_id", "subject_id"], columns="dim_name", values="dim_intensity", aggfunc="mean", fill_value=0).reset_index()
    every = [c for c in wide.columns if c not in ("stimulus_id", "subject_id")]
    sets = {"13 features used in the notebook (12 appraisal dimensions + arousal)": [c for c in every if c != "valence"],
            "12 appraisal dimensions (arousal and valence excluded)": [c for c in every if c not in ("valence", "arousal")]}
    for label, columns in sets.items():
        frame = pd.DataFrame({"iss": isc(wide, "stimulus_id", columns)}).join(avg[["valence", "arousal"]])
        frame["pos"] = (frame.valence > 5).astype(int)
        a, b = frame.iss[frame.pos == 1], frame.iss[frame.pos == 0]
        adjusted = smf.ols("iss ~ pos + arousal", frame).fit()
        add("D", f"similarity, {label}", f"positive {a.mean():.3f}, negative {b.mean():.3f}, d = {(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2):.2f}; "
                                          f"valence coefficient adjusted for arousal {adjusted.params['pos']:.3f} (t = {adjusted.tvalues['pos']:.1f})")
    category_wide = cats.pivot_table(index=["stimulus_id", "subject_id"], columns="cat_name", values="cat_intensity", aggfunc="mean", fill_value=0).reset_index()
    columns = [c for c in category_wide.columns if c not in ("stimulus_id", "subject_id")]
    frame = pd.DataFrame({"iss": isc(category_wide, "stimulus_id", columns),
                          "raters": category_wide.groupby("stimulus_id").size(),
                          "endorsed": cats.groupby(["stimulus_id", "subject_id"]).size().groupby("stimulus_id").mean()}).join(avg[["valence", "arousal"]]).dropna()
    frame["pos"] = (frame.valence > 5).astype(int)
    raw_fit = smf.ols("iss ~ pos", frame).fit()
    adjusted = smf.ols("iss ~ pos + arousal + endorsed + raters", frame).fit()
    add("D", "emotion-category similarity, valence coefficient raw / adjusted for arousal, categories endorsed per rater, rater count",
        f"{raw_fit.params['pos']:.3f} (t = {raw_fit.tvalues['pos']:.1f}) / {adjusted.params['pos']:.3f} (t = {adjusted.tvalues['pos']:.1f})")


def study3():
    ratings = pd.read_csv(HERE / "IAPS_rating.csv", index_col=0)
    images = sorted(ratings.Image.unique())
    info, cluster = category_info(images)
    n = ratings.groupby("Image").size().iloc[0]
    add("E", "raters per image; expected in-sample R2 of OLS with 13 predictors under no relationship", f"{n}; {13 / (n - 1):.2f}")

    def cross_validated(factory):
        out = []
        for image in images:
            g = ratings[ratings.Image == image]
            X, Y = g[FEATURES].values, g[TARGETS].values
            P = np.zeros_like(Y, dtype=float)
            for train, test in LeaveOneOut().split(X):
                P[test] = factory(len(train)).fit(X[train], Y[train]).predict(X[test])
            out.append(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        return np.array(out)
    variants = {"PLS 13 components (the notebook; equivalent to OLS)": lambda m: PLSRegression(n_components=min(13, m - 1)),
                "PLS 1 component": lambda m: PLSRegression(n_components=1), "PLS 2 components": lambda m: PLSRegression(n_components=2),
                "ridge alpha = 100": lambda m: Ridge(alpha=100), "ridge alpha = 1000": lambda m: Ridge(alpha=1000)}
    for label, factory in variants.items():
        x = cross_validated(factory)
        compare("E", f"leave-one-subject-out CV R2, {label} (images with R2 > 0: {(x > 0).sum()}/136)", x, info, cluster)
    nested, chosen = [], []
    for image in images:
        g = ratings[ratings.Image == image]
        X, Y = g[FEATURES].values, g[TARGETS].values
        P, picked = nested_prediction(X, Y)
        nested.append(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        chosen.append(picked)
    x, chosen = np.array(nested), np.array(chosen)
    compare("E", f"nested CV R2, complexity chosen inside training raters (mean-only or PLS 1-5; images with R2 > 0: {(x > 0).sum()}/136)", x, info, cluster)
    pos = (info.Valence == "pos").values
    add("E", "nested CV: share of folds choosing the mean-only model, positive / negative images",
        f"{(chosen[pos] == 0).mean():.2f} / {(chosen[~pos] == 0).mean():.2f}")
    coefficients = []
    for image in images:
        g = ratings[ratings.Image == image]
        coef = PLSRegression(n_components=13).fit(g[FEATURES], g[TARGETS]).coef_
        coef = coef if coef.shape[0] == 13 else coef.T
        coefficients.append((coef.mean(), np.abs(coef).mean()))
    coefficients = np.array(coefficients)
    for j, label in enumerate(("signed mean coefficient (as reported)", "mean absolute coefficient")):
        a, b = coefficients[pos, j], coefficients[~pos, j]
        add("E", label, f"positive {a.mean():.4f}, negative {b.mean():.4f}, t = {stats.ttest_ind(a, b)[0]:+.2f}")


def study3_cowen_keltner():
    ratings = pd.read_csv(HERE / "IAPS_rating.csv", index_col=0)
    result = cowen_keltner(ratings)
    for label in ("positive", "negative"):
        add("F", f"{label} images (n = {result['n'][label]}): explainable variance (mean of six emotions)", f"{result['explainable'][label]:.2f}")
        add("F", f"{label} images: leave-one-out explained variance / explainable variance, linear; k-NN (k = {result['k'][label]})",
            f"{result['linear'][label]:.3f}; {result['knn'][label]:.3f}")
    for which, name in (("linear", "linear regression"), ("knn", "k-nearest-neighbor regression")):
        lo, hi = result[which]["interval"]
        add("F", f"{name}: positive minus negative normalized explained variance (category bootstrap, 95% interval)",
            f"{result[which]['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    for step in (valence_labels, recovered_ratings, raw_ratings, study2, study3, study3_cowen_keltner):
        step()
    pd.DataFrame(rows, columns=["Section", "Check", "Value"]).to_csv(HERE / "figs" / "robustness_studies1to3.csv", index=False)
