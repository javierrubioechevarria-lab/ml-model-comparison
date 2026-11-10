"""Common functions for Phases 4 to 6 (explained line by line in the Phase 3 notebook)."""
import numpy as np  # numpy: numerical computing with arrays
import polars as pl  # polars: reads the data table
from pathlib import Path  # Path: handles file paths
from scipy import stats  # stats: Student's t distribution
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score, brier_score_loss, log_loss  # metrics
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold  # data splitting and folds


def load_data():  # reads the file and repeats the train/test split of Phases 2 and 3
    path = Path("datasets/breast_cancer.csv.zst")  # the data file is in the datasets folder
    data = pl.read_csv(path)  # read the compressed file
    data = data.with_columns((1 - pl.col("target")).alias("malignant")).drop("target")  # malignant = 1 (in the file benign = 1)
    columns = [c for c in data.columns if c != "malignant"]  # names of the 30 features
    X = data.select(columns).to_numpy()  # 569 x 30 matrix
    y = data["malignant"].to_numpy()  # diagnosis (1 = malignant)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)  # the usual split
    return X_train, X_test, y_train, y_test, columns, path  # return everything


def make_folds(X_train, y_train):  # the 25 folds of Phase 3 (5 chunks x 5 repetitions)
    cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)  # same seed => same folds
    return list(cross_validation.split(X_train, y_train))  # list of (training rows, validation rows)


def compute_metrics(y_true, score, threshold=0.5):  # all the metrics for a set of predictions
    pred = (score >= threshold).astype(int)  # "malignant" if the probability exceeds the threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()  # the four counts
    p = np.clip(score, 1e-6, 1 - 1e-6)  # avoids log(0) in the log-loss
    return {  # dictionary with all the metrics
        "mcc": matthews_corrcoef(y_true, pred),  # main metric
        "recall": tp / (tp + fn),  # fraction of malignant tumours detected
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,  # of those we called malignant, how many were
        "f1": f1_score(y_true, pred, zero_division=0),  # harmonic mean of precision and recall
        "accuracy": (tp + tn) / len(y_true),  # fraction of correct predictions
        "cost": (5 * fn + fp) / len(y_true),  # average cost per patient (5:1 assumption)
        "fn": int(fn), "fp": int(fp),  # false negatives and false positives
        "auc": roc_auc_score(y_true, score),  # quality of the probability ranking
        "brier": brier_score_loss(y_true, p),  # squared error of the probabilities
        "logloss": log_loss(y_true, p, labels=[0, 1]),  # punishes confident wrong probabilities
    }  # end of the dictionary


def best_threshold(y_true, score):  # finds the threshold that minimizes the cost
    thresholds = np.linspace(0.01, 0.99, 99)  # 99 candidate thresholds
    costs = [compute_metrics(y_true, score, t)["cost"] for t in thresholds]  # cost of each threshold
    return float(thresholds[int(np.argmin(costs))])  # the threshold with the lowest cost


def corrected_ttest(a, b, n_train, n_val):  # Nadeau-Bengio t-test to compare two models
    d = np.asarray(a) - np.asarray(b)  # fold-by-fold difference
    k = len(d)  # number of folds
    t = d.mean() / np.sqrt((1 / k + n_val / n_train) * d.var(ddof=1))  # t corrected for overlapping folds
    p = 2 * stats.t.sf(abs(t), df=k - 1)  # two-sided p-value
    return d.mean(), t, p  # mean difference, t and p-value


def summarize(values):  # mean, standard deviation and 95% interval of a per-fold metric
    values = np.asarray(values)  # array with the values
    std = values.std(ddof=1)  # standard deviation across folds
    half_ci = stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))  # half-width of the 95% CI
    return values.mean(), std, half_ci  # return the three
