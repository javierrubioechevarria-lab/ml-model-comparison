# %% Cell 1: first look at the data
import polars as pl  # polars: reads and manipulates the data table
from pathlib import Path  # Path: handles file paths

path = Path("datasets/breast_cancer.csv.zst")  # the data file is in the datasets folder
data = pl.read_csv(path)  # read the compressed file
print("Rows (tumours):", data.height)  # number of rows
print("Columns:", data.width)  # 30 features + 1 target column
print("Target column:", "target")  # the diagnosis
print("Note: in the file target=1 is BENIGN; in Phase 2 we recode it as malignant=1")  # important warning
data.head(3)  # first 3 rows
