import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# color pallette for pretty data viz <3
PETAL   = "#E8749A"   # deep sakura pink
PETAL_L = "#F4B8CE"  # light petal
LEAF    = "#5C8A55"   # spring green
SKY     = "#A8C8E8"   # pale sky
BARK    = "#7A5C4A"   # tree bark
INK     = "#2A1F1A"   # near-black
CREAM   = "#FDF6EE"   # warm off-white
ACCENT  = "#C84B6F"   # deep rose accent

"""
# downloading the latest version (returns a directory)
path = kagglehub.dataset_download("ryanglasnapp/japanese-cherry-blossom-data")
print("Path to dataset files:", path)
"""

# loading the dataset
df = pd.read_csv("sakura_full_bloom_dates.csv")

"""
# loadingthe dataset
full_bloom_raw = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ryanglasnapp/japanese-cherry-blossom-data",
    "sakura_full_bloom_dates.csv",  #I chose FULL bloom dates (not first bloom)
)"""

#----DATA PROCESSING----

# years for which the data was collected
year_col = [c for c in df.columns if str(c).isdigit()]

# day of the year for each year (in each site)
all_days = df[year_col].apply(lambda col: pd.to_datetime(col, errors="coerce").dt.dayofyear)
all_days.index = df["Site Name"]

# mean of full bloom per year
year_mean = all_days.mean(axis=0)
year_mean.index = year_mean.index.astype(int)

#dropping the years that didn't have data recorded (if a site had no observations)
all_days = all_days[~np.isnan(all_days)]

#----PLOTTING ALL DATA POINTS----
plt.figure()
plt.title("Distribution of cherry blossom's full-bloom days", fontsize=14, fontfamily='serif')
plt.xlabel("Year")
plt.ylabel("Day of the Year")
for site in all_days.index:
    plt.scatter(all_days.columns.astype(int), all_days.loc[site], color=PETAL, alpha= 0.15, s=10)
plt.show()

#----PLOTTING THE HISTOGRAM----
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year')
plt.ylabel('Count')
plt.xlim(min(all_days), max(all_days))
plt.hist(year_mean, color=PETAL, edgecolor='white')

plt.show()