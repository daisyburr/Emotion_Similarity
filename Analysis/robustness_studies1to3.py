"""Robustness checks for Studies 1-3 (supplementary; every figure quoted for these checks in the paper comes from here).

A. Study 2: similarity with arousal excluded from the feature set (12 appraisal dimensions); valence effect on similarity
   adjusted for arousal, for both feature spaces.
B. Study 3: leave-one-subject-out cross-validated R2 under PLS with 1 or 2 components, ridge regression, and a valence-blind
   nested cross-validation that chooses model complexity inside the training raters (Table S5).
C. Study 3 following Cowen & Keltner (2017, PNAS): leave-one-out cross-validated linear and k-nearest-neighbor regression of the
   emotion ratings on the appraisal ratings across images (rater means), run separately within positive and negative images;
   explained variance divided by the explainable variance (split-half reliability of each emotion rating across images,
   Spearman-Brown corrected); k chosen from 1-50 as the value with the greatest average explained variance; the positive-minus-negative
   difference is bootstrapped by resampling stimulus categories within each valence (Table S1), with the across-image and
   across-rater variances behind the comparison.
Category-level p-values use the exact cluster-label permutation in fmri/supplement_analyses.py.

Usage: python robustness_studies1to3.py
Output: Analysis/figs/robustness_studies1to3.csv
"""
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
from study3_transformation import FEATURES, TARGETS, cowen_keltner, image_table, nested_prediction  # noqa: E402

rows = []


def cohens_d(a, b):
    """Cohen's d with the pooled standard deviation (as in the notebook)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def add(section, check, value):
    rows.append((section, check, value))
    print(f"[{section}] {check}: {value}", flush=True)


def compare(section, label, x, info, cluster):
    pos = (info.Valence == "pos").values
    a, b = x[pos], x[~pos]
    t, p = stats.ttest_ind(a, b)
    d = cohens_d(a, b)
    _, pe, _, n_clusters, n_relabel = S.category_level(x[:, None], info, cluster)
    add(section, label, f"positive {a.mean():.3f}, negative {b.mean():.3f}, d = {d:+.2f}, image-level p = {p:.2g}, "
                        f"category-level exact p = {pe[0]:.2g} ({n_clusters} clusters)")


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
        add("A", "Study 2 checks", "skipped (Cowen & Keltner files not present)")
        return
    dims = pd.read_csv(ck / "indiv_resp_dimensions.csv")
    avg = pd.read_csv(ck / "averages_dimensions.csv").set_index("stimulus_id")
    wide = dims.pivot_table(index=["stimulus_id", "subject_id"], columns="dim_name", values="dim_intensity", aggfunc="mean", fill_value=0).reset_index()
    every = [c for c in wide.columns if c not in ("stimulus_id", "subject_id")]
    sets = {"13 features used in the notebook (12 appraisal dimensions + arousal)": [c for c in every if c != "valence"],
            "12 appraisal dimensions (arousal and valence excluded)": [c for c in every if c not in ("valence", "arousal")]}
    for label, columns in sets.items():
        frame = pd.DataFrame({"iss": isc(wide, "stimulus_id", columns)}).join(avg[["valence", "arousal"]])
        frame["pos"] = (frame.valence > 5).astype(int)
        a, b = frame.iss[frame.pos == 1], frame.iss[frame.pos == 0]
        adjusted = smf.ols("iss ~ pos + arousal", frame).fit()
        add("A", f"similarity, {label}", f"positive {a.mean():.3f}, negative {b.mean():.3f}, d = {cohens_d(a, b):.2f}; "
                                          f"valence coefficient adjusted for arousal {adjusted.params['pos']:.3f} (t = {adjusted.tvalues['pos']:.1f})")


def study3():
    ratings = pd.read_csv(HERE / "IAPS_rating.csv", index_col=0)
    images = sorted(ratings.Image.unique())
    info, cluster = image_table(images)

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
    variants = {"PLS 1 component": lambda m: PLSRegression(n_components=1), "PLS 2 components": lambda m: PLSRegression(n_components=2),
                "ridge alpha = 100": lambda m: Ridge(alpha=100), "ridge alpha = 1000": lambda m: Ridge(alpha=1000)}
    for label, factory in variants.items():
        x = cross_validated(factory)
        compare("B", f"leave-one-subject-out CV R2, {label} (images with R2 > 0: {(x > 0).sum()}/136)", x, info, cluster)
    nested, chosen = [], []
    for image in images:
        g = ratings[ratings.Image == image]
        X, Y = g[FEATURES].values, g[TARGETS].values
        P, picked = nested_prediction(X, Y)
        nested.append(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        chosen.append(picked)
    x, chosen = np.array(nested), np.array(chosen)
    compare("B", f"nested CV R2, complexity chosen inside training raters (mean-only or PLS 1-5; images with R2 > 0: {(x > 0).sum()}/136)", x, info, cluster)
    pos = (info.Valence == "pos").values
    add("B", "nested CV: share of folds choosing the mean-only model, positive / negative images",
        f"{(chosen[pos] == 0).mean():.2f} / {(chosen[~pos] == 0).mean():.2f}")


def study3_cowen_keltner():
    ratings = pd.read_csv(HERE / "IAPS_rating.csv", index_col=0)
    result = cowen_keltner(ratings)
    for label in ("positive", "negative"):
        add("C", f"{label} images (n = {result['n'][label]}): explainable variance (mean of six emotions)", f"{result['explainable'][label]:.2f}")
        add("C", f"{label} images: leave-one-out explained variance / explainable variance, linear; k-NN (k = {result['k'][label]})",
            f"{result['linear'][label]:.3f}; {result['knn'][label]:.3f}")
    images = sorted(ratings.Image.unique())
    info, _ = image_table(images)
    means = ratings.groupby("Image")[FEATURES + TARGETS].mean().reindex(images)
    within = ratings.groupby("Image")[TARGETS].var().mean(1).reindex(images)
    for label, code in (("positive", "pos"), ("negative", "neg")):
        mask = (info.Valence == code).values
        add("C", f"{label} images: across-image variance of mean appraisal ratings; of mean emotion ratings; across-rater variance of emotion ratings within image",
            f"{means.loc[mask, FEATURES].var().mean():.0f}; {means.loc[mask, TARGETS].var().mean():.0f}; {within[mask].mean():.0f}")
    for which, name in (("linear", "linear regression"), ("knn", "k-nearest-neighbor regression")):
        lo, hi = result[which]["interval"]
        add("C", f"{name}: positive minus negative normalized explained variance (category bootstrap, 95% interval)",
            f"{result[which]['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    for step in (study2, study3, study3_cowen_keltner):
        step()
    pd.DataFrame(rows, columns=["Section", "Check", "Value"]).to_csv(HERE / "figs" / "robustness_studies1to3.csv", index=False)
