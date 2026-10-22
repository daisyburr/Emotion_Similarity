"""Study 3: appraisal-to-emotion transformation, rater-level and image-level analyses.

nested_cv_r2        For each image, predict every rater's six emotion ratings from their 13 appraisal ratings and score held-out raters
                    (leave-one-subject-out). Model complexity (mean-only, or PLS with 1-5 components) is chosen by 6-fold
                    cross-validation inside the training raters, so the choice never sees the held-out rater or the valence label.
cowen_keltner       Following Cowen & Keltner (2017, PNAS): leave-one-out cross-validated linear and k-nearest-neighbor regression of the
                    emotion ratings on the appraisal ratings across images (rater means), separately within positive and negative
                    images; explained variance divided by the explainable variance (split-half reliability of each emotion rating
                    across images, Spearman-Brown corrected); k chosen from 1-50 as the value with the greatest average explained
                    variance. The positive-minus-negative difference is bootstrapped by resampling stimulus categories within each
                    valence.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

SEED = 0
FEATURES = ["Narrative_Rating", "Immoral_Rating", "Pleasant_Rating", "Threatened_Rating", "Avoid_Rating",
            "SociallyAcceptable_Rating", "Interact_Rating", "Contamination_Rating", "Suffering_Rating",
            "Affection_Rating", "Desire_Rating", "Astonished_Rating", "Relate_Rating"]
TARGETS = [f"{e}_Rating" for e in ["Surprise", "Sadness", "Joy", "Fear", "Anger", "Disgust"]]
CATEGORY_VALENCE = {"AA": "neg", "AC": "pos", "AT": "pos", "CA": "pos", "CH": "pos", "CO": "neg", "FA": "neg", "FO": "pos",
                    "HC": "neg", "HM": "neg", "HV": "neg", "NC": "neg", "NI": "neg", "NM": "neg", "NN": "neg", "PC": "pos",
                    "PI": "pos", "PM": "pos", "PN": "pos", "SI": "pos", "SX": "pos", "TH": "neg"}


def image_table(images):
    """Valence and stimulus category per image, and the category-by-valence cluster label used for category-level tests."""
    info = pd.DataFrame({"Valence": [CATEGORY_VALENCE[i[:2]] for i in images], "Category": [i[:2] for i in images]}, index=images)
    cluster = info.Category + info.Valence.map({"pos": "+", "neg": "-"})
    return info, cluster


class MeanModel:
    def fit(self, X, Y):
        self.mean = Y.mean(0)
        return self

    def predict(self, X):
        return np.tile(self.mean, (len(X), 1))


def make_model(k):
    return MeanModel() if k == 0 else PLSRegression(n_components=k, max_iter=5000)


def nested_prediction(X, Y, ks=(0, 1, 2, 3, 4, 5)):
    predictions, chosen = np.zeros_like(Y, dtype=float), []
    for test in range(len(X)):
        train = np.arange(len(X)) != test
        Xt, Yt = X[train], Y[train]
        error = {k: 0.0 for k in ks}
        for a, b in KFold(6, shuffle=True, random_state=SEED).split(Xt):
            for k in ks:
                error[k] += ((Yt[b] - make_model(k).fit(Xt[a], Yt[a]).predict(Xt[b])) ** 2).sum()
        best = min(error, key=error.get)
        chosen.append(best)
        predictions[test] = make_model(best).fit(Xt, Yt).predict(X[test:test + 1])
    return predictions, chosen


def nested_cv_r2(ratings):
    """Per-image nested cross-validated R2 (pooled over the six emotions) and the share of folds that chose the mean-only model."""
    images = sorted(ratings.Image.unique())
    info, _ = image_table(images)
    out = []
    for image in images:
        g = ratings[ratings.Image == image]
        X, Y = g[FEATURES].values, g[TARGETS].values
        P, chosen = nested_prediction(X, Y)
        out.append({"Image": image, "Valence": info.Valence[image],
                    "nested_CV_R2": 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum(),
                    "share_mean_only": float(np.mean(np.array(chosen) == 0))})
    return pd.DataFrame(out)


def _loo_linear(X, Y):
    Z = np.column_stack([np.ones(len(X)), X])
    H = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    return Y - (Y - H @ Y) / (1 - np.diag(H))[:, None]


def _loo_knn(X, Y, k):
    Xs = (X - X.mean(0)) / X.std(0)
    distance = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(distance, np.inf)
    return Y[np.argsort(distance, 1)[:, :k]].mean(1)


def _explained(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def cowen_keltner(ratings, n_boot=2000):
    images = sorted(ratings.Image.unique())
    subjects = sorted(ratings.Subject.unique())
    wide = lambda column: ratings.pivot(index="Subject", columns="Image", values=column).reindex(index=subjects, columns=images).values
    predictors = np.stack([wide(c) for c in FEATURES], -1)
    targets = np.stack([wide(c) for c in TARGETS], -1)
    info, _ = image_table(images)
    valence, category = info.Valence.values, info.Category.values
    rng = np.random.default_rng(SEED)
    half = len(subjects) // 2

    fits = {}
    for label, mask in (("positive", valence == "pos"), ("negative", valence == "neg")):
        X, Y = predictors.mean(0)[mask], targets.mean(0)[mask]
        reliability = []
        for _ in range(200):
            perm = rng.permutation(len(subjects))
            a, b = targets[perm[:half]][:, mask].mean(0), targets[perm[half:2 * half]][:, mask].mean(0)
            r = np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(Y.shape[1])])
            reliability.append(2 * r / (1 + r))
        explainable = np.mean(reliability, 0)
        ks = range(1, min(50, len(X) - 1) + 1)
        scores = {k: _explained(Y, _loo_knn(X, Y, k)) for k in ks}
        best = max(ks, key=lambda k: scores[k].mean())
        fits[label] = {"mask": mask, "Y": Y, "linear": _loo_linear(X, Y), "knn": _loo_knn(X, Y, best), "explainable": explainable, "k": best}

    def normalized(label, which, index):
        fit = fits[label]
        Y, P = fit["Y"][index], fit[which][index]
        return ((1 - ((Y - P) ** 2).sum(0) / ((Y - fit["Y"].mean(0)) ** 2).sum(0)) / fit["explainable"]).mean()

    groups = {label: {c: np.where(category[fits[label]["mask"]] == c)[0] for c in np.unique(category[fits[label]["mask"]])} for label in fits}
    result = {"n": {label: int(fits[label]["mask"].sum()) for label in fits},
              "explainable": {label: float(fits[label]["explainable"].mean()) for label in fits}, "k": {label: fits[label]["k"] for label in fits}}
    for which in ("linear", "knn"):
        values = {label: float(normalized(label, which, np.arange(fits[label]["Y"].shape[0]))) for label in fits}
        draws = []
        for _ in range(n_boot):
            pair = []
            for label in ("positive", "negative"):
                keys = list(groups[label])
                pick = rng.choice(len(keys), len(keys))
                pair.append(normalized(label, which, np.concatenate([groups[label][keys[i]] for i in pick])))
            draws.append(pair[0] - pair[1])
        result[which] = {**values, "difference": values["positive"] - values["negative"],
                         "interval": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))}
    return result
