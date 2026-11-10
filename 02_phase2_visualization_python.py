# %% Cell 1: import the libraries
import polars as pl  # polars: reads and manipulates the data table
import numpy as np  # numpy: numerical computing with arrays
import matplotlib.pyplot as plt  # matplotlib: draws the plots
import seaborn as sns  # seaborn: nicer statistical plots
from pathlib import Path  # Path: handles folder and file paths
from scipy.stats import skew  # skew: measures the asymmetry of a distribution
from sklearn.model_selection import train_test_split  # train_test_split: separates train and test

# %% Cell 2: load the data file
path = Path("datasets/breast_cancer.csv.zst")  # the data file is in the datasets folder
data = pl.read_csv(path)  # read the compressed file with polars
figures_dir = path.parent.parent / "figures" / "02_visualization"  # folder where we will save the plots
figures_dir.mkdir(parents=True, exist_ok=True)  # create it if it does not exist
print("Rows:", data.height)  # number of tumours
print("Columns:", data.width)  # number of columns (30 features + the diagnosis)

# %% Cell 3: recode the diagnosis
data = data.with_columns((1 - pl.col("target")).alias("malignant"))  # in the file target=1 is BENIGN; we create malignant=1 so that "positive" is the malignant tumour
data = data.drop("target")  # drop the old column to avoid confusion
print(data["malignant"].value_counts().sort("malignant"))  # check how many of each type (0 = benign, 1 = malignant)

# %% Cell 4: split train and test BEFORE looking at the data
indices = np.arange(data.height)  # list of row numbers 0, 1, 2, ...
train_indices, test_indices = train_test_split(  # split the rows into two groups
    indices,  # the rows to distribute
    test_size=0.20,  # 20% is kept for the final test
    random_state=42,  # fixed seed: the split is always the same
    stratify=data["malignant"].to_numpy(),  # keeps the same share of malignant tumours in both groups
)
train = data[train_indices.tolist()]  # training table: the only one we will look at
test = data[test_indices.tolist()]  # test table: not touched until the end
print("Train:", train.height, "rows")  # training size
print("Test:", test.height, "rows")  # test size

# %% Cell 5: first rows
train.head(5)  # show the first 5 rows of the training set

# %% Cell 6: distribution of the diagnosis
counts = train.group_by("malignant").len().sort("malignant")  # count how many tumours of each type
print(counts)  # show the count table
total = counts["len"].sum()  # total number of training rows
for kind, amount in zip(counts["malignant"], counts["len"]):  # loop over each type
    print(f"malignant={kind}: {amount} cases ({amount / total:.1%})")  # show the percentage of each type
plt.figure(figsize=(6, 4))  # create a 6x4 inch figure
plt.bar(["Benign", "Malignant"], counts["len"].to_list(), color=["#4C72B0", "#C44E52"])  # bars: blue benign, red malignant
plt.title("Class distribution (training set)")  # plot title
plt.ylabel("Number of cases")  # vertical axis name
plt.savefig(figures_dir / "01_classes.png", bbox_inches="tight")  # save the plot as an image
plt.show()  # show it on screen

# %% Cell 7: list of features
columns = [c for c in train.columns if c != "malignant"]  # all columns except the diagnosis
print("Number of features:", len(columns))  # must be 30
for i, name in enumerate(columns):  # loop over the names with their number
    print(i + 1, name)  # print them numbered

# %% Cell 8: choose 6 features to look at closely
selected = ["mean_area", "area_error", "mean_concavity", "worst_perimeter", "mean_smoothness", "texture_error"]  # each one illustrates a different issue
fig, axes = plt.subplots(2, 3, figsize=(14, 7))  # grid of 2 rows and 3 columns
for ax, column in zip(axes.ravel(), selected):  # one feature per panel
    values = train[column].to_numpy()  # convert the column to a numpy array
    sns.histplot(values, bins=25, kde=True, ax=ax)  # histogram with the density curve
    ax.set_title(column)  # title: feature name
    ax.set_xlabel("Value")  # horizontal axis name
    ax.set_ylabel("Frequency")  # vertical axis name
plt.tight_layout()  # adjust spacing so nothing overlaps
plt.savefig(figures_dir / "02_histograms.png", bbox_inches="tight")  # save the image
plt.show()  # show it

# %% Cell 9: scales (are all features in the same range?)
medians = {c: float(np.median(train[c].to_numpy())) for c in columns}  # median of each feature
for column in selected:  # show min, median and max of the 6 chosen features
    values = train[column]  # current column
    print(f"{column}: min {values.min():.4f} | median {medians[column]:.4f} | max {values.max():.4f}")  # one line per feature
largest = max(medians, key=medians.get)  # feature with the largest median
smallest = min(medians, key=medians.get)  # feature with the smallest median
print(f"\nLargest median: {largest} = {medians[largest]:.1f}")  # its value
print(f"Smallest median: {smallest} = {medians[smallest]:.4f}")  # its value
print(f"The largest is {medians[largest] / medians[smallest]:,.0f} times the smallest")  # how many times larger

# %% Cell 10: missing values and duplicates
nulls = train.null_count().row(0)  # nulls in each column (one row of results)
print("Total null values:", sum(nulls))  # sum of all nulls
total_nan = sum(int(np.isnan(train[c].to_numpy()).sum()) for c in columns)  # NaN (invalid number) in each column
print("Total NaN values:", total_nan)  # sum of all NaN
print("Duplicated rows:", int(train.is_duplicated().sum()))  # repeated rows

# %% Cell 11: outliers (box plots)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))  # 2x3 grid
for ax, column in zip(axes.ravel(), selected):  # one feature per panel
    values = train[column].to_numpy()  # convert the column to numpy
    sns.boxplot(x=values, ax=ax)  # box: the isolated points are possible outliers
    ax.set_title(column)  # panel title
    ax.set_xlabel("Value")  # horizontal axis name
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "03_outliers.png", bbox_inches="tight")  # save the image
plt.show()  # show it

# %% Cell 12: count the outliers with the IQR rule
def count_outliers(values):  # function: flags which values are outliers
    q1 = np.quantile(values, 0.25)  # first quartile (25% of the data below)
    q3 = np.quantile(values, 0.75)  # third quartile (75% of the data below)
    iqr = q3 - q1  # interquartile range: width of the box
    outside = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)  # True if the value falls outside the 1.5·IQR rule
    return outside  # array of True/False, one per row

for column in selected:  # count in the 6 chosen features
    print(column, ":", int(count_outliers(train[column].to_numpy()).sum()), "possible outliers")  # how many there are

outside_all = np.column_stack([count_outliers(train[c].to_numpy()) for c in columns])  # True/False table for the 30 features
has_any = outside_all.any(axis=1)  # True if the tumour has at least one outlier
malignant = train["malignant"].to_numpy()  # diagnosis of each row (1 = malignant)
print(f"\nTumours with any outlier: {has_any.sum()} of {len(has_any)} ({has_any.mean():.0%})")  # total and percentage
print(f"  among malignant: {has_any[malignant == 1].mean():.0%}")  # percentage within malignant tumours
print(f"  among benign: {has_any[malignant == 0].mean():.0%}")  # percentage within benign tumours

# %% Cell 13: skewness of the distributions
skews = {c: float(skew(train[c].to_numpy())) for c in columns}  # skewness of each feature (0 = symmetric)
very_skewed = [c for c in columns if skews[c] > 1]  # those with a long right tail
print("Features with skewness > 1:", len(very_skewed), "of", len(columns))  # how many
most_skewed = sorted(skews, key=skews.get, reverse=True)[:5]  # the 5 most skewed
for c in most_skewed:  # show them
    print(f"  {c}: {skews[c]:.2f}")  # name and skewness
before = np.mean([skews[c] for c in very_skewed])  # mean skewness of those features
after = np.mean([skew(np.log1p(train[c].to_numpy())) for c in very_skewed])  # the same after applying log1p
print(f"Mean skewness of those features: before {before:.2f}, after log1p {after:.2f}")  # compare

# %% Cell 14: effect of log1p on one feature
values = train["area_error"].to_numpy()  # take area_error, very skewed
fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # two plots side by side
sns.histplot(values, bins=30, ax=axes[0], color="#55A868")  # original histogram
axes[0].set_title(f"area_error original (skewness {skew(values):.1f})")  # title with its skewness
sns.histplot(np.log1p(values), bins=30, ax=axes[1], color="#8172B2")  # histogram after log1p
axes[1].set_title(f"area_error with log1p (skewness {skew(np.log1p(values)):.1f})")  # title with its skewness
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "04_log1p.png", bbox_inches="tight")  # save the image
plt.show()  # show it

# %% Cell 15: correlation of each feature with the diagnosis
X = train.select(columns).to_numpy()  # feature matrix (rows x 30)
y = train["malignant"].to_numpy()  # diagnosis vector (1 = malignant)
matrix = np.corrcoef(np.column_stack([X, y]), rowvar=False)  # correlation between all columns, including the diagnosis
target_correlations = matrix[:-1, -1]  # last column: correlation of each feature with the diagnosis
order = np.argsort(np.abs(target_correlations))[::-1]  # features sorted from highest to lowest |correlation|
print("Features most related to the diagnosis (+ = more malignant):")  # title
for i in order[:10]:  # the first 10
    print(f"  {columns[i]}: {target_correlations[i]:+.3f}")  # name and signed correlation
print("\nThe 3 least related:")  # title
for i in order[-3:]:  # the last 3
    print(f"  {columns[i]}: {target_correlations[i]:+.3f}")  # name and correlation

# %% Cell 16: heat map of the most important ones
indices = list(order[:10]) + [len(columns)]  # positions of the top 10 + the diagnosis
names = [columns[i] for i in order[:10]] + ["Malignant"]  # their names
reduced_matrix = matrix[np.ix_(indices, indices)]  # sub-matrix with only those columns
plt.figure(figsize=(11, 8))  # large figure so the numbers fit
sns.heatmap(reduced_matrix, xticklabels=names, yticklabels=names, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)  # heat map with numbers
plt.title("Correlations between the 10 features most related to the diagnosis, and the diagnosis")  # title
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "05_top10_correlations.png", bbox_inches="tight")  # save the image
plt.show()  # show it

# %% Cell 17: how many features are repeated among themselves?
feature_correlation = np.corrcoef(X, rowvar=False)  # correlation between the 30 features (without the diagnosis)
upper_part = np.triu_indices(len(columns), k=1)  # positions above the diagonal: each pair only once
high_pairs = np.abs(feature_correlation[upper_part]) > 0.9  # True if the pair has |correlation| > 0.9
print("Feature pairs with |r| > 0.9:", int(high_pairs.sum()), "of", len(high_pairs))  # how many pairs
j = columns.index("mean_radius")  # position of mean_radius
print("mean_radius with mean_perimeter:", round(float(feature_correlation[j, columns.index("mean_perimeter")]), 3))  # radius and perimeter
print("mean_radius with mean_area:", round(float(feature_correlation[j, columns.index("mean_area")]), 3))  # radius and area

# %% Cell 18: the two best features against the diagnosis
fig, axes = plt.subplots(1, 2, figsize=(11, 5))  # two plots side by side
for ax, i in zip(axes, order[:2]):  # the two most related features
    sns.boxplot(x=y, y=X[:, i], ax=ax)  # box of the feature by diagnosis (0 = benign, 1 = malignant)
    ax.set_xticks([0, 1])  # ticks at 0 and 1
    ax.set_xticklabels(["Benign", "Malignant"])  # name of each type
    ax.set_title(columns[i])  # title: feature name
    ax.set_xlabel("Diagnosis")  # horizontal axis
    ax.set_ylabel("Value")  # vertical axis
plt.tight_layout()  # adjust spacing
plt.savefig(figures_dir / "06_feature_vs_diagnosis.png", bbox_inches="tight")  # save the image
plt.show()  # show it

# %% Cell 19: does a single feature separate almost everything?
best = columns[order[0]]  # the feature most related to the diagnosis
values = train[best].to_numpy()  # its values
thresholds = np.linspace(values.min(), values.max(), 200)  # 200 test thresholds between the minimum and the maximum
hits = [np.mean((values > t) == (y == 1)) for t in thresholds]  # fraction of correct answers if we say "malignant" when the value exceeds the threshold
print(f"With a single threshold on {best} we get {max(hits):.1%} of the cases right")  # best result
print(f"Always predicting 'benign' gets {np.mean(y == 0):.1%}")  # what the baseline gets
