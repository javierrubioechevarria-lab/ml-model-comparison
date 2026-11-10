# %% Cell 1: import the libraries
import os  # os: configure the environment
import time  # time: measure how long things take
import warnings  # warnings: silence repetitive warnings
import numpy as np  # numpy: numerical computing with arrays
import polars as pl  # polars: result tables
import matplotlib.pyplot as plt  # matplotlib: draws the plots
from sklearn.base import clone  # clone: copies a model without training
from sklearn.metrics import make_scorer, matthews_corrcoef  # MCC as the selection criterion
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, validation_curve  # tuning and validation curves with cross-validation
from utils_python import load_data, make_folds, compute_metrics, best_threshold  # functions from Phase 3
from models_python import learners  # the five learners and their grids (Phase 4)
os.environ["PYTHONWARNINGS"] = "ignore"  # parallel processes should not print warnings either
warnings.filterwarnings("ignore")  # silence repetitive convergence warnings

# %% Cell 2: load data, folds and learners
X_train, X_test, y_train, y_test, columns, path = load_data()  # same 455/114 split
folds = make_folds(X_train, y_train)  # the 25 OUTER folds of Phase 3
results_dir = path.parent.parent / "results"  # folder for tables
figures_dir = path.parent.parent / "figures" / "05_tuning"  # folder for plots
results_dir.mkdir(parents=True, exist_ok=True)  # create it if it does not exist
figures_dir.mkdir(parents=True, exist_ok=True)  # create it if it does not exist
print("Outer folds:", len(folds), "| learners:", list(learners))  # reminder
print("The test set (", len(y_test), "rows) is still untouched")  # reminder

# %% Cell 3: ONE fold step by step · split the outer fold
fold_number = 0  # choose the first outer fold
train_idx, val_idx = folds[fold_number]  # its training and validation rows
X_tr, y_tr = X_train[train_idx], y_train[train_idx]  # data to learn from and tune on (364 rows)
X_va, y_va = X_train[val_idx], y_train[val_idx]  # data to evaluate on (91 rows): NOT touched until the end
print("Fold training set:", len(y_tr), "| outer validation:", len(y_va))  # sizes

# %% Cell 4: ONE fold step by step · choose hyperparameters (INNER validation)
family, pipeline, grid = learners["svm"]  # example with the analogizer (SVM)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 INNER folds, only inside the training part
search = GridSearchCV(clone(pipeline), grid, scoring=make_scorer(matthews_corrcoef), cv=inner, n_jobs=-1)  # tries every combination and scores it with MCC
search.fit(X_tr, y_tr)  # the outer validation does NOT take part
print("Combinations tried:", len(search.cv_results_["params"]), "| fits:", len(search.cv_results_["params"]) * 3)  # 80 x 3
print("Best inner MCC:", round(search.best_score_, 3))  # the best score reached inside the training part
for name, value in search.best_params_.items():  # show the best combination
    print(" ", name, "=", "log1p" if hasattr(value, "func") else value)  # name and chosen value

# %% Cell 5: ONE fold step by step · choose the threshold by cost
final_model = search.best_estimator_  # the best model, already retrained with the whole fold training set
out_of_fold = cross_val_predict(clone(final_model), X_tr, y_tr, cv=inner, method="predict_proba")[:, 1]  # probabilities of tumours that each model did NOT see
threshold = best_threshold(y_tr, out_of_fold)  # threshold that minimizes the cost, using only the training part
print("Chosen threshold:", threshold)  # usually lower than 0.5

# %% Cell 6: ONE fold step by step · score the outer validation (only once)
prob_va = final_model.predict_proba(X_va)[:, 1]  # probability of malignant for the 91 validation tumours
for name, t in [("threshold 0.5", 0.5), ("cost threshold", threshold)]:  # the two thresholds
    m = compute_metrics(y_va, prob_va, t)  # metrics with that threshold
    print(f"{name}: MCC {m['mcc']:.3f} | recall {m['recall']:.3f} | FN {m['fn']} | FP {m['fp']} | cost {m['cost']:.3f}")  # result

# %% Cell 7: the same recipe as a function (to repeat it on the 25 folds and the 5 learners)
def label(value):  # makes hyperparameter values readable
    if isinstance(value, str):  # "passthrough" means "untransformed"
        return "no log1p" if value == "passthrough" else value  # readable name
    return "log1p" if hasattr(value, "func") else str(value)  # log1p or the value as text

def tune_fold(key, fold_number):  # tunes one learner on one outer fold
    family, pipeline, grid = learners[key]  # pipeline and grid of the learner
    train_idx, val_idx = folds[fold_number]  # rows of the fold
    X_tr, y_tr, X_va, y_va = X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx]  # split
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # inner validation
    start = time.perf_counter()  # start the clock
    search = GridSearchCV(clone(pipeline), grid, scoring=make_scorer(matthews_corrcoef), cv=inner, n_jobs=-1).fit(X_tr, y_tr)  # 1) choose hyperparameters
    out_of_fold = cross_val_predict(clone(search.best_estimator_), X_tr, y_tr, cv=inner, method="predict_proba")[:, 1]  # 2) out-of-sample probabilities
    threshold = best_threshold(y_tr, out_of_fold)  # 2) cost threshold
    seconds = time.perf_counter() - start  # total tuning time
    prob_va = search.best_estimator_.predict_proba(X_va)[:, 1]  # 3) score the outer validation
    row = {"learner": key, "fold": fold_number, "threshold": threshold, "inner_mcc": float(search.best_score_), "seconds": seconds}  # fold data
    for prefix, t in [("t05", 0.5), ("tcost", threshold)]:  # metrics with the two thresholds
        row.update({f"{prefix}_{k}": v for k, v in compute_metrics(y_va, prob_va, t).items()})  # add each metric
    row["params"] = "; ".join(f"{k.replace('model__', '').replace('estimator__', '')}={label(v)}" for k, v in search.best_params_.items())  # chosen hyperparameters
    scores = [{"learner": key, "fold": fold_number, "row": int(i), "y": int(y), "score": float(p)} for i, y, p in zip(val_idx, y_va, prob_va)]  # one line per tumour
    return row, scores  # return results

test_row, _ = tune_fold("bayes", 0)  # check the function with a fast learner
print("Naive Bayes, fold 0 ->", test_row["params"], "| threshold", test_row["threshold"])  # result of the check

# %% Cell 8: full tuning (5 learners x 25 folds)
rows, all_scores = [], []  # here we accumulate the results
total_start = time.perf_counter()  # overall clock
for key in learners:  # loop over the five learners
    start = time.perf_counter()  # the learner's clock
    for n in range(len(folds)):  # loop over the 25 outer folds
        row, scores = tune_fold(key, n)  # tune this fold
        rows.append(row)  # store the row
        all_scores.extend(scores)  # store the scores
    print(f"{key:12s} finished in {time.perf_counter() - start:5.1f} s")  # progress
print(f"Total time: {(time.perf_counter() - total_start) / 60:.1f} min")  # total time
results = pl.from_dicts(rows, infer_schema_length=None)  # table with one row per (learner, fold)
outer_scores = pl.from_dicts(all_scores)  # table with one row per (learner, fold, tumour)
results.write_csv(results_dir / "outer_folds.csv")  # save for Phase 6
outer_scores.write_csv(results_dir / "outer_scores.csv")  # save for Phase 6
print("Saved in:", results_dir)  # where

# %% Cell 9: how much does the tuning cost?
for key, (family, _, grid) in learners.items():  # loop over the five
    sub = results.filter(pl.col("learner") == key)  # its 25 rows
    n_comb = int(np.prod([len(v) for v in grid.values()]))  # combinations in its grid
    print(f"{family:13s} {n_comb:3d} configurations | {sub['seconds'].sum() / 60:5.2f} min in total | {sub['seconds'].mean():5.1f} s per fold | mean inner MCC {sub['inner_mcc'].mean():.3f}")  # summary

# %% Cell 10: which hyperparameters were chosen? (counts)
chosen = {}  # dictionary: learner -> {hyperparameter -> list of chosen values}
for key in learners:  # loop over the five
    chosen[key] = {}  # empty dictionary for the learner
    for text in results.filter(pl.col("learner") == key)["params"]:  # one string per fold
        for pair in text.split("; "):  # split "name=value"
            name, value = pair.split("=", 1)  # name and value
            chosen[key].setdefault(name, []).append(value)  # note it down
    for name, values in chosen[key].items():  # show how many times each value came up
        unique_values, counts = np.unique(values, return_counts=True)  # count
        counts_text = ", ".join(f"{v}: {c}" for v, c in sorted(zip(unique_values, counts), key=lambda t: -t[1]))  # sorted text
        print(f"{key:12s} {name:20s} -> {counts_text}")  # one line per hyperparameter

# %% Cell 11: plot of the hyperparameter stability
plot_keys = [("tree", "model__max_depth"), ("mlp", "model__alpha"), ("bayes", "model__var_smoothing"), ("svm", "model__estimator__C"), ("svm", "model__estimator__gamma"), ("evolutionary", "log1p")]  # the most informative ones
fig, axes = plt.subplots(2, 3, figsize=(14, 7))  # 2x3 grid
for ax, (key, full_name) in zip(axes.ravel(), plot_keys):  # one plot per hyperparameter
    name = full_name.replace("model__", "").replace("estimator__", "")  # short name
    options = [label(o) for o in learners[key][2][full_name]]  # grid values, in their order
    counts = [chosen[key][name].count(o) for o in options]  # how many folds chose each value
    ax.bar(range(len(options)), counts, color="#4C72B0")  # bars in grid order
    ax.set_xticks(range(len(options)))  # one tick per value
    ax.set_xticklabels(options, rotation=45)  # value names
    ax.set_title(f"{key}: {name}")  # title
    ax.set_ylabel("Folds (out of 25)")  # vertical axis
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "01_chosen_hyperparameters.png", bbox_inches="tight")  # save
plt.show()  # show

# %% Cell 12: was the edge of the grid chosen many times?
for key, (family, _, grid) in learners.items():  # loop over the five
    for full_name, options in grid.items():  # loop over its hyperparameters
        if len(options) < 3:  # with fewer than 3 options "edge" makes no sense
            continue  # skip
        name = full_name.replace("model__", "").replace("estimator__", "")  # short name
        edges = {label(options[0]), label(options[-1])}  # first and last value of the grid
        times = sum(v in edges for v in chosen[key][name])  # folds in which an edge was chosen
        warning = "  <-- check" if times / 25 > 0.4 else ""  # flag if it is frequent
        print(f"{key:12s} {name:20s} edge in {times:2d} of 25 folds{warning}")  # result

# %% Cell 13: which thresholds did each learner choose?
fig, ax = plt.subplots(figsize=(8, 4))  # one figure
threshold_data = [results.filter(pl.col("learner") == key)["threshold"].to_numpy() for key in learners]  # the 25 thresholds of each learner
ax.boxplot(threshold_data, tick_labels=list(learners))  # one box per learner
ax.axhline(0.5, color="red", linestyle="--", label="default threshold (0.5)")  # line of the usual threshold
ax.set_ylabel("Threshold chosen by cost")  # vertical axis
ax.set_title("Threshold that minimizes the cost, in each of the 25 folds")  # title
ax.legend()  # legend
plt.savefig(figures_dir / "02_thresholds.png", bbox_inches="tight")  # save
plt.show()  # show
for key in learners:  # numeric summary
    u = results.filter(pl.col("learner") == key)["threshold"]  # thresholds of the learner
    print(f"{key:12s} mean threshold {u.mean():.2f} | min {u.min():.2f} | max {u.max():.2f} | folds with threshold <= 0.05: {int((u <= 0.05).sum())}")  # summary

# %% Cell 14: validation curves (does more complexity always help?)
curves = [("tree", "model__max_depth", [1, 2, 3, 4, 5, 6, 8, 10], "max_depth"), ("mlp", "model__alpha", [1e-3, 1e-2, 1e-1, 1, 10, 100], "alpha (regularization)"), ("svm", "model__estimator__gamma", [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1], "gamma")]  # learner, hyperparameter, values to try, axis name
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # one panel per learner
for ax, (key, parameter, values, axis_name) in zip(axes, curves):  # loop over the three learners
    pipeline = learners[key][1]  # the learner's pipeline with its default hyperparameters
    train_mcc, valid_mcc = validation_curve(clone(pipeline), X_train, y_train, param_name=parameter, param_range=values, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring=make_scorer(matthews_corrcoef), n_jobs=-1)  # MCC on the training part and on the validation part of 5 folds
    ax.plot(range(len(values)), train_mcc.mean(axis=1), "o-", label="training")  # curve on the data the model saw
    ax.plot(range(len(values)), valid_mcc.mean(axis=1), "o-", label="validation")  # curve on data it did not see
    ax.set_xticks(range(len(values)))  # one tick per value
    ax.set_xticklabels([str(v) for v in values], rotation=45)  # value names
    ax.set_xlabel(axis_name)  # horizontal axis
    ax.set_ylabel("MCC (5-fold CV on the 455 training rows)")  # vertical axis
    ax.set_title(key)  # title
    ax.legend()  # legend
    print(key, "| validation MCC by value:", {v: round(float(m), 3) for v, m in zip(values, valid_mcc.mean(axis=1))}, "| training MCC at the last value:", round(float(train_mcc.mean(axis=1)[-1]), 3))  # numbers
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "03_validation_curves.png", bbox_inches="tight")  # save
plt.show()  # show
