import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import geom

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
#print(all_days[all_days < 70].stack())

# mean of full bloom per year
year_mean = all_days.mean(axis=0)
year_mean.index = year_mean.index.astype(int)


#----PLOTTING ALL DATA POINTS----
plt.figure()
plt.title("Distribution of cherry blossom's full-bloom days", fontsize=14, fontfamily='serif')
plt.xlabel("Year", fontsize=12, fontfamily='serif')
plt.ylabel("Day of the Year", fontsize=12, fontfamily='serif')
for site in all_days.index:
    plt.scatter(all_days.columns.astype(int), all_days.loc[site], color=PETAL, alpha= 0.15, s=10)
plt.show()

#----PLOTTING THE HISTOGRAM----
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year', fontsize=12, fontfamily='serif')
plt.ylabel('Count', fontsize=12, fontfamily='serif')
no_bins = int(max(all_days)) - int(min(all_days))
plt.hist(year_mean, bins=no_bins, color=PETAL_L, edgecolor='white')
plt.show()


#because I noticed that the southern part of Japan has very different blooming dates than the rest of Japan, I decided to clean the data further and exclude those data points to investigate the distribution for "mainland" Japan
#to do that, I first printed all the locations: print(all_days.index.tolist()), then manually excluded Islands, by asking AI to classify which location is where on the map, link to conversation: https://claude.ai/share/b978ebe4-da5d-4a47-897d-e9af627c7481
#print(all_days.index.tolist())
excluded_locations = [
  "Wakkanai", "Rumoi", "Asahikawa", "Abashiri", "Sapporo", "Iwamizawa",
  "Obihiro", "Kushiro", "Nemuro", "Muroran", "Urakawa", "Esashi",
  "Hakodate", "Kutchan", "Monbetsu", "Hiroo",

  "Naze", "Yonaguni Island", "Ishigaki Island", "Miyakojima",
  "Kumejima", "Naha", "Nago", "Iriomote Island", "Minami Daito Island"
]
all_days_mainland = all_days[~all_days.index.isin(excluded_locations)]
year_mean_mainland = all_days_mainland.mean(axis=0)
#----PLOTTING THE HISTOGRAM 2 (of means across ALL locations)----
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates (Central Japan)", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year', fontsize=12, fontfamily='serif')
plt.ylabel('Count', fontsize=12, fontfamily='serif')
no_bins = int(max(all_days)) - int(min(all_days))
plt.hist(year_mean_mainland, bins=no_bins, color=PETAL_L, edgecolor='white')
plt.axvline(x=91, color=SKY, linewidth=1, label='April 1st')   # day 91, assuming lap years don't exist for simplicity :DD I experimented also with adding March 1st and May 1st but they were just too far from the data
plt.legend()
plt.show()

#----PLOTTING THE HISTOGRAM 3 (of means across mainland locations)---- with bell-curve
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates (Central Japan)", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year', fontsize=12, fontfamily='serif')
plt.ylabel('Count', fontsize=12, fontfamily='serif')
no_bins = (int(max(all_days)) - int(min(all_days)))//5 #grouping data points into 5 so that they better match the normal curve
#plt.hist(year_mean_mainland, bins=no_bins, color=PETAL_L, edgecolor='white')
plt.axvline(x=91, color=SKY, linewidth=1, label='April 1st')   # day 91, assuming lap years don't exist for simplicity :DD I experimented also with adding March 1st and May 1st but they were just too far from the data
plt.legend()
from scipy.stats import norm
mu, std = norm.fit(year_mean_mainland)
x = np.linspace(year_mean_mainland.min(), year_mean_mainland.max(), 100)
counts, bins, _ = plt.hist(year_mean_mainland, bins=no_bins, color=PETAL_L, edgecolor='white')
bin_width = bins[1] - bins[0]
plt.plot(x, norm.pdf(x, mu, std) * len(year_mean_mainland) *bin_width, color=LEAF, lw=2, label='Normal fit')
from scipy.stats import skewnorm, gamma, lognorm

# Gamma
a, loc, scale = gamma.fit(year_mean_mainland)
plt.plot(x, gamma.pdf(x, a, loc, scale) * len(year_mean_mainland) * bin_width, color=ACCENT, label='Gamma Fit')
plt.show()

#----PLOTTING THE HISTOGRAM 4 (of individual datapoints across mainland locations)----
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates (Central Japan), INDIVIDUAL DATAPOINTS", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year', fontsize=12, fontfamily='serif')
plt.ylabel('Count', fontsize=12, fontfamily='serif')
processed_datapoints = all_days_mainland.to_numpy().ravel()
processed_datapoints = processed_datapoints[np.isfinite(processed_datapoints)]

no_bins = int(max(processed_datapoints)) - int(min(processed_datapoints))
counts, bins, _ = plt.hist(processed_datapoints, bins=no_bins, color=PETAL_L, edgecolor='white')
bin_width = bins[1] - bins[0]
plt.show()


# getting the parameters to fit a geometric distribution:
mean = processed_datapoints.mean()
print(f"Mean of geometric is: {mean}")
var = processed_datapoints.var()
print(f"Var of geometric is: {var}")
likelihoods = {}
first_datapoint = int(processed_datapoints.min()) -1
print(f"Datapoints start at: {first_datapoint}")
p = 1 / (mean - first_datapoint)
print(f"Probability of success each day is: {p}")
log_likelihood_geom = sum(np.log(geom.pmf(x, p, loc=first_datapoint)) for x in processed_datapoints)

#----PLOTTING THE HISTOGRAM 5: fitting geometric to individual datapoints across mainland locations)----
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates (Central Japan), GEOMETRIC DISTRIBUTION, INDIVIDUAL DATAPOINTS", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year', fontsize=12, fontfamily='serif')
plt.ylabel('Count', fontsize=12, fontfamily='serif')
processed_datapoints = all_days_mainland.to_numpy().ravel()
processed_datapoints = processed_datapoints[np.isfinite(processed_datapoints)]

no_bins = int(max(processed_datapoints)) - int(min(processed_datapoints))
counts, bins, _ = plt.hist(processed_datapoints, bins=no_bins, color=PETAL_L, edgecolor='white')
bin_width = bins[1] - bins[0]
# Geometric
x_days = np.arange(int(processed_datapoints.min()), int(processed_datapoints.max()))
plt.plot(x_days, geom.pmf(x_days, p, loc=first_datapoint) * len(processed_datapoints),
         color=ACCENT, lw=2, label='Geometric fit')
plt.legend()
plt.show()

print("mean:", processed_datapoints.mean())
print("min:", processed_datapoints.min())
print("loc:", first_datapoint)
print("p:", p)
print("sample pmf at peak day:", geom.pmf(97, p, loc=first_datapoint))
print("scaled:", geom.pmf(97, p, loc=first_datapoint) * len(processed_datapoints))


#----PLOTTING THE HISTOGRAM 6: fitting negative binomial to individual datapoints across mainland locations)----
from scipy.stats import nbinom
plt.figure()
plt.title("Japanese Cherry Blossom - Full Bloom Dates (Central Japan), NEGATIVE BINOM INDIVIDUAL DATAPOINTS", fontsize=14, fontfamily='serif')
plt.xlabel('Day of the year', fontsize=12, fontfamily='serif')
plt.ylabel('Count', fontsize=12, fontfamily='serif')
processed_datapoints = all_days_mainland.to_numpy().ravel()
processed_datapoints = processed_datapoints[np.isfinite(processed_datapoints)]
no_bins = int(max(processed_datapoints)) - int(min(processed_datapoints))
counts, bins, _ = plt.hist(processed_datapoints, bins=no_bins, color=PETAL_L, edgecolor='white')
# fit negative binomial
shifted = processed_datapoints - first_datapoint  # shift first
mean_shifted = shifted.mean()
var_shifted = shifted.var()

# method of moments estimates
p_nb = mean_shifted / var_shifted
n_nb = mean_shifted * p_nb / (1 - p_nb)

x_days = np.arange(int(processed_datapoints.min()), int(processed_datapoints.max()))
bin_width = bins[1] - bins[0]

print(f"stdev: {processed_datapoints.std()}")
print(f"median: {np.median(processed_datapoints)}")


def normal_pmf(day, mu, std):
    p = norm.cdf(day + 0.5, mu, std) - norm.cdf(day - 0.5, mu, std)
    return max(p, 1e-300)  # never return exactly 0

def nbinom_pmf(day, n, p, loc):
    prob = nbinom.pmf(day, n, p, loc=loc)
    return max(prob, 1e-300)

loc = int(processed_datapoints.min()) - 1
# normal parameters
mu_normal = processed_datapoints.mean()
std_normal = processed_datapoints.std()

# nbinom parameters  
shifted = processed_datapoints - first_datapoint
p_nb = shifted.mean() / shifted.var()
n_nb = shifted.mean() * p_nb / (1 - p_nb)

# normal
plt.plot(x_days, [normal_pmf(x, mu_normal, std_normal) * len(processed_datapoints) for x in x_days],
         color=LEAF, lw=2, label='Discretized Normal fit')

# nbinom
plt.plot(x_days, [nbinom_pmf(x, n_nb, p_nb, loc) * len(processed_datapoints) for x in x_days],
         color=ACCENT, lw=2, label='Negative Binomial fit')


# then use correct variables in PMFs
log_lik_normal = sum(np.log(normal_pmf(x, mu_normal, std_normal)) for x in processed_datapoints)
log_lik_nbinom = sum(np.log(nbinom_pmf(x, n_nb, p_nb, loc)) for x in processed_datapoints)

log_num_normal = log_lik_normal + np.log(0.5)
log_num_nbinom = log_lik_nbinom + np.log(0.5)

# log-sum-exp trick
max_log = max(log_num_normal, log_num_nbinom)
log_p_D = max_log + np.log(np.exp(log_num_normal - max_log) + np.exp(log_num_nbinom - max_log))

P_normal = np.exp(log_num_normal - log_p_D)
P_nbinom  = np.exp(log_num_nbinom - log_p_D)

print("P(normal | data):", P_normal)
print("P(nbinom | data):", P_nbinom)
plt.legend()
plt.show()


print("nbinom pmf at day 97:", nbinom_pmf(97, n_nb, p_nb, loc))
print("normal pmf at day 97:", normal_pmf(97, mu_normal, std_normal))
print("nbinom pmf at day 145:", nbinom_pmf(145, n_nb, p_nb, loc))
print("normal pmf at day 145:", normal_pmf(145, mu_normal, std_normal))