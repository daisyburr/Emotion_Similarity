"""Study 3: appraisal-to-emotion prediction, rater level.

nested_cv_r2        For each image, predict every rater's six emotion ratings from their 13 appraisal ratings and score held-out raters
                    (leave-one-subject-out). Model complexity (mean-only, or PLS with 1-5 components) is chosen by 6-fold
                    cross-validation inside the training raters, so the choice never sees the held-out rater or the valence label.
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
