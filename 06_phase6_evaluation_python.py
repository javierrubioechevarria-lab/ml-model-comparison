# %% Cell 1: import the libraries
import os  # os: configure the environment
import time  # time: measure times
import warnings  # warnings: silence repetitive warnings
import numpy as np  # numpy: numerical computing with arrays
import polars as pl  # polars: result tables
import matplotlib.pyplot as plt  # matplotlib: draws the plots
from scipy.stats import binomtest  # binomial test (exact McNemar and Wilson interval)
from sklearn.base import clone  # clone: copies a model without training
from sklearn.calibration import calibration_curve  # reliability curve
from sklearn.linear_model import LogisticRegression  # logistic regression (comparison with the evolutionary learner)
from sklearn.metrics import confusion_matrix, make_scorer, matthews_corrcoef  # metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # tuning with cross-validation
from utils_python import load_data, make_folds, compute_metrics, best_threshold, corrected_ttest, summarize  # functions from Phase 3
from models_python import learners, build, EvolutionaryClassifier  # learners from Phase 4
os.environ["PYTHONWARNINGS"] = "ignore"  # no warnings in the parallel processes
warnings.filterwarnings("ignore")  # no repetitive warnings

# %% Cell 2: load the data and the Phase 5 results
X_train, X_test, y_train, y_test, columns, path = load_data()  # same 455/114 split
folds = make_folds(X_train, y_train)  # the 25 outer folds
results_dir = path.parent.parent / "results"  # where Phase 5 saved its tables
figures_dir = path.parent.parent / "figures" / "06_evaluation"  # plots of this phase
figures_dir.mkdir(parents=True, exist_ok=True)  # create it if it does not exist
results = pl.read_csv(results_dir / "outer_folds.csv")  # metrics per learner and outer fold
scores = pl.read_csv(results_dir / "outer_scores.csv")  # probability of each validation tumour
keys = list(learners)  # ["tree", "mlp", "evolutionary", "bayes", "svm"]
names = {"tree": "Tree", "mlp": "MLP", "evolutionary": "Evolutionary", "bayes": "Naive Bayes", "svm": "SVM"}  # names for the tables
n_train, n_val = len(folds[0][0]), len(folds[0][1])  # 364 and 91 (for the corrected t-test)
print("Result rows:", results.height, "(5 learners x 25 folds)")  # check
print("The test set (", len(y_test), "rows) has not been used yet")  # reminder

# %% Cell 3: results table in the outer cross-validation (threshold 0.5)
print(f"{'Learner':12s} {'MCC':>14s} {'Recall':>14s} {'AUC':>14s} {'Brier':>14s} {'Log-loss':>14s}")  # header
for key in keys:  # one line per learner
    sub = results.filter(pl.col("learner") == key).sort("fold")  # its 25 folds
    cells = []  # the formatted metrics go here
    for metric in ["mcc", "recall", "auc", "brier", "logloss"]:  # the metrics of the table
        mean, std, _ = summarize(sub[f"t05_{metric}"].to_numpy())  # mean and standard deviation across folds
        cells.append(f"{mean:.3f} ± {std:.3f}")  # text "mean ± standard deviation"
    print(f"{names[key]:12s} " + " ".join(f"{c:>14s}" for c in cells))  # one table line
print("Baseline (always benign / always malignant): MCC 0 | recall 0 / 1 | AUC 0.5 | cost 1.868 / 0.626")  # from Phase 3

# %% Cell 4: dispersion across folds
fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # two plots side by side
for ax, (prefix, metric, title) in zip(axes, [("t05", "mcc", "MCC per fold (threshold 0.5)"), ("tcost", "cost", "Cost per fold (cost threshold)")]):  # MCC and cost
    box_data = [results.filter(pl.col("learner") == k).sort("fold")[f"{prefix}_{metric}"].to_numpy() for k in keys]  # 25 values per learner
    ax.boxplot(box_data, tick_labels=[names[k] for k in keys], showmeans=True)  # boxes with the mean
    for i, values in enumerate(box_data):  # individual points
        ax.scatter(np.random.default_rng(0).normal(i + 1, 0.05, len(values)), values, s=12, alpha=0.5, color="#4C72B0")  # one point per fold
    ax.set_title(title)  # title
axes[1].axhline(0.626, color="red", linestyle="--", label="always malignant (0.626)")  # cost floor
axes[1].legend()  # legend
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "01_dispersion_comparison.png", bbox_inches="tight")  # save
plt.show()  # show

# %% Cell 5: are the differences real? Corrected t-test + Holm
def holm(p):  # Holm correction for multiple comparisons
    p = np.asarray(p)  # p-values as an array
    order = np.argsort(p)  # from smallest to largest
    adjusted = np.empty(len(p))  # the adjusted p-values go here
    running_max = 0.0  # the adjusted value can never drop below the previous one
    for rank, i in enumerate(order):  # loop from smallest to largest
        running_max = max(running_max, (len(p) - rank) * p[i])  # multiply by the comparisons that remain
        adjusted[i] = min(1.0, running_max)  # the adjusted p-value cannot exceed 1
    return adjusted  # return

fold_mcc = {k: results.filter(pl.col("learner") == k).sort("fold")["t05_mcc"].to_numpy() for k in keys}  # MCC of the 25 folds of each learner
pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]  # the 10 possible pairs
t_rows = []  # results of each pair
for a, b in pairs:  # loop over the pairs
    diff, t, p = corrected_ttest(fold_mcc[a], fold_mcc[b], n_train, n_val)  # corrected t-test
    t_rows.append((a, b, diff, p))  # store
holm_p = holm([r[3] for r in t_rows])  # correct the 10 p-values at once
print(f"{'Pair':26s} {'MCC diff.':>9s} {'p':>8s} {'Holm p':>8s}")  # header
for (a, b, diff, p), hp in zip(t_rows, holm_p):  # one line per pair
    flag = "  <-- significant" if hp < 0.05 else ""  # flag if it survives Holm
    print(f"{names[a] + ' vs ' + names[b]:26s} {diff:+9.3f} {p:8.4f} {hp:8.4f}{flag}")  # result

# %% Cell 6: effect of choosing the threshold by cost
print(f"{'Learner':12s} {'cost 0.5':>10s} {'cost thr.':>10s} {'saving':>8s} {'FN 0.5→t':>12s} {'FP 0.5→t':>12s}")  # header
for key in keys:  # one line per learner
    sub = results.filter(pl.col("learner") == key)  # its 25 folds
    c05, ct = sub["t05_cost"].mean(), sub["tcost_cost"].mean()  # mean costs
    print(f"{names[key]:12s} {c05:10.3f} {ct:10.3f} {1 - ct / c05:8.0%} {sub['t05_fn'].mean():5.2f}→{sub['tcost_fn'].mean():4.2f} {sub['t05_fp'].mean():6.2f}→{sub['tcost_fp'].mean():4.2f}")  # summary

# %% Cell 7: calibration of the probabilities
fig, ax = plt.subplots(figsize=(6, 6))  # square figure
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")  # ideal diagonal
for key in keys:  # one curve per learner
    sub = scores.filter(pl.col("learner") == key)  # scores of its 25 folds
    frequency, mean_prob = calibration_curve(sub["y"].to_numpy(), sub["score"].to_numpy(), n_bins=10)  # real fraction of malignant tumours per probability bin
    ax.plot(mean_prob, frequency, marker="o", label=names[key])  # the learner's curve
ax.set_xlabel("Predicted probability of malignant")  # horizontal axis
ax.set_ylabel("Real fraction of malignant tumours")  # vertical axis
ax.set_title("Reliability curve (outer validation)")  # title
ax.legend()  # legend
plt.savefig(figures_dir / "02_calibration.png", bbox_inches="tight")  # save
plt.show()  # show
for key in keys:  # how many scores are extreme
    p = scores.filter(pl.col("learner") == key)["score"].to_numpy()  # its scores
    print(f"{names[key]:12s} extreme scores (<0.02 or >0.98): {np.mean((p < 0.02) | (p > 0.98)):.0%}")  # percentage

# %% Cell 8: fix the "best" learner BEFORE looking at the test set
mean_mcc = {k: fold_mcc[k].mean() for k in keys}  # mean MCC of each learner in the outer CV
best = max(mean_mcc, key=mean_mcc.get)  # the one with the highest mean MCC
print("Mean MCC in the outer CV:", {names[k]: round(float(v), 3) for k, v in mean_mcc.items()})  # all of them
print("Best learner fixed BEFORE the test set:", names[best])  # the chosen one

# %% Cell 9: final refit on the whole training set and evaluation on the TEST set (only once)
def refit_final(key):  # same recipe as Phase 5, but with the 455 rows
    family, pipeline, grid = learners[key]  # pipeline and grid
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # inner validation
    search = GridSearchCV(clone(pipeline), grid, scoring=make_scorer(matthews_corrcoef), cv=inner, n_jobs=-1).fit(X_train, y_train)  # choose hyperparameters
    out_of_fold = cross_val_predict(clone(search.best_estimator_), X_train, y_train, cv=inner, method="predict_proba")[:, 1]  # out-of-sample probabilities
    return search.best_estimator_, best_threshold(y_train, out_of_fold)  # final model and cost threshold

test_prob, final_threshold = {}, {}  # probabilities on the test set and threshold of each learner
for key in keys:  # the five learners
    model, final_threshold[key] = refit_final(key)  # refit (the test set plays no role)
    test_prob[key] = model.predict_proba(X_test)[:, 1]  # the ONLY time the test set is scored
pl.DataFrame({"y": y_test, **test_prob}).write_csv(results_dir / "test_scores.csv")  # save the probabilities
print(f"{'Learner':12s} {'thresh.':>7s} {'MCC':>6s} {'recall':>7s} {'FN':>3s} {'FP':>3s} {'cost':>6s}")  # header
for key in keys:  # results with threshold 0.5 and with the cost threshold
    for threshold_label, t in [("0.5", 0.5), (f"{final_threshold[key]:.2f}", final_threshold[key])]:  # the two thresholds
        m = compute_metrics(y_test, test_prob[key], t)  # metrics on the test set
        print(f"{names[key]:12s} {threshold_label:>7s} {m['mcc']:6.3f} {m['recall']:7.3f} {m['fn']:3d} {m['fp']:3d} {m['cost']:6.3f}")  # one line

# %% Cell 10: are the differences significant on the test set? (exact McNemar)
correct = {k: (test_prob[k] >= 0.5).astype(int) == y_test for k in keys}  # True if the learner gets each tumour right (threshold 0.5)
for other in keys:  # compare the best with each of the others
    if other == best:  # do not compare it with itself
        continue  # skip
    only_best = int(np.sum(correct[best] & ~correct[other]))  # tumours only the best gets right
    only_other = int(np.sum(~correct[best] & correct[other]))  # tumours only the other gets right
    p = binomtest(min(only_best, only_other), only_best + only_other, 0.5).pvalue if only_best + only_other > 0 else 1.0  # exact McNemar = binomial test on the disagreements
    print(f"{names[best]} vs {names[other]:12s} only the best is right: {only_best} | only the other is right: {only_other} | p = {p:.3f}")  # result

# %% Cell 11: uncertainty of the test set (Wilson interval of the recall)
malignant_count = int(y_test.sum())  # malignant tumours in the test set
detected = int(np.sum((test_prob[best] >= 0.5) & (y_test == 1)))  # malignant tumours detected by the best
ci = binomtest(detected, malignant_count).proportion_ci(confidence_level=0.95, method="wilson")  # Wilson interval
print(f"{names[best]}: detects {detected} of {malignant_count} malignant | recall {detected / malignant_count:.3f} | 95% CI [{ci.low:.3f}, {ci.high:.3f}]")  # result

# %% Cell 12: diagnosis of the best learner (confusion matrix and cost curve)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))  # two plots
matrix = confusion_matrix(y_test, (test_prob[best] >= 0.5).astype(int), labels=[0, 1])  # confusion matrix on the test set
axes[0].imshow(matrix, cmap="Blues")  # draw the matrix
for i in range(2):  # loop over rows
    for j in range(2):  # and columns
        axes[0].text(j, i, matrix[i, j], ha="center", va="center", fontsize=18)  # number in each cell
axes[0].set_xticks([0, 1])  # horizontal ticks
axes[0].set_xticklabels(["Pred. benign", "Pred. malignant"])  # names
axes[0].set_yticks([0, 1])  # vertical ticks
axes[0].set_yticklabels(["Real benign", "Real malignant"])  # names
axes[0].set_title(f"{names[best]} on the test set (threshold 0.5)")  # title
sub = scores.filter(pl.col("learner") == best)  # scores of the best in the outer CV
thresholds = np.linspace(0.01, 0.99, 99)  # thresholds to try
costs = [compute_metrics(sub["y"].to_numpy(), sub["score"].to_numpy(), t)["cost"] for t in thresholds]  # cost of each threshold in the outer CV
axes[1].plot(thresholds, costs)  # cost curve
axes[1].axvline(0.5, color="gray", linestyle="--", label="threshold 0.5")  # default threshold
axes[1].axvline(final_threshold[best], color="red", linestyle="--", label=f"chosen threshold ({final_threshold[best]:.2f})")  # cost threshold
axes[1].set_xlabel("Threshold")  # horizontal axis
axes[1].set_ylabel("Average cost per patient (outer CV)")  # vertical axis
axes[1].set_title("Cost against threshold")  # title
axes[1].legend()  # legend
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "03_best_learner_diagnosis.png", bbox_inches="tight")  # save
plt.show()  # show

# %% Cell 13: error analysis on the test set
failures = np.sum([~correct[k] for k in keys], axis=0)  # how many learners fail on each test tumour
print("Tumours that everybody gets right:", int(np.sum(failures == 0)), "of", len(y_test))  # the majority
print("Tumours that at least one learner gets wrong:", int(np.sum(failures > 0)))  # the difficult ones
print("Tumours that all five learners get wrong:", int(np.sum(failures == 5)))  # the impossible ones
for k in keys:  # errors per learner
    print(f"  {names[k]:12s} fails on {int((~correct[k]).sum())} tumours")  # how many
i_per, i_con = columns.index("worst_perimeter"), columns.index("worst_concave_points")  # positions of the two best features
fig, ax = plt.subplots(figsize=(8, 6))  # figure
easy = failures == 0  # tumours without errors
ax.scatter(X_test[easy, i_per], X_test[easy, i_con], c=np.where(y_test[easy] == 1, "#C44E52", "#4C72B0"), s=15, alpha=0.4)  # background: red malignant, blue benign
ax.scatter(X_test[~easy, i_per], X_test[~easy, i_con], c="black", s=40 + 40 * failures[~easy], marker="x")  # crosses: the more learners fail, the bigger
ax.set_xlabel("worst_perimeter")  # horizontal axis
ax.set_ylabel("worst_concave_points")  # vertical axis
ax.set_title("Test set: errors (crosses) fall in the overlap zone")  # title
plt.savefig(figures_dir / "04_errors.png", bbox_inches="tight")  # save
plt.show()  # show
difficult = np.where(~easy)[0]  # positions of the difficult tumours
print(f"worst_perimeter of the difficult ones: from {X_test[difficult, i_per].min():.0f} to {X_test[difficult, i_per].max():.0f}")  # range
print(f"worst_concave_points of the difficult ones: from {X_test[difficult, i_con].min():.2f} to {X_test[difficult, i_con].max():.2f}")  # range

# %% Cell 14: evolutionary hypothesis, does it tie with a gradient-based linear model?
logistic = build(True, LogisticRegression(max_iter=1000))  # logistic regression with scaling (no hyperparameter tuning)
logistic_mcc, logistic_times = [], []  # MCC and time of each fold
for train_idx, val_idx in folds:  # the same 25 folds
    start = time.perf_counter()  # clock
    model = clone(logistic).fit(X_train[train_idx], y_train[train_idx])  # train on the fold
    logistic_times.append(time.perf_counter() - start)  # fit time
    logistic_mcc.append(compute_metrics(y_train[val_idx], model.predict_proba(X_train[val_idx])[:, 1])["mcc"])  # MCC on validation
diff, t, p = corrected_ttest(fold_mcc["evolutionary"], np.array(logistic_mcc), n_train, n_val)  # evolutionary - logistic
print(f"Evolutionary MCC {fold_mcc['evolutionary'].mean():.3f} | logistic MCC {np.mean(logistic_mcc):.3f} | difference {diff:+.3f} | corrected p = {p:.2f}")  # comparison
start = time.perf_counter()  # evolutionary clock
build(True, EvolutionaryClassifier()).fit(X_train[folds[0][0]], y_train[folds[0][0]])  # one evolutionary fit (40 individuals x 100 generations)
evo_time = time.perf_counter() - start  # its time
print(f"Time of one fit: evolutionary {evo_time:.2f} s | logistic {np.mean(logistic_times) * 1000:.0f} ms | the evolutionary learner is {evo_time / np.mean(logistic_times):.0f} times slower")  # cost of blind search
