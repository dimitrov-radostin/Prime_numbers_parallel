import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sizes = np.array([10000,50000,100000,500000,1000000,5000000,10000000,50000000], dtype=float)
# avgs  = np.array([0.001231533,0.006292,0.0126222,0.067908,0.1378932,0.817865867,1.0891333,4.862121933], dtype=float)
# stds  = np.array([0.000165468,0.000271151,0.000351521,0.002367902,0.023103607,0.031549985,0.173016477,0.267644543], dtype=float)

df = pd.read_csv("run_data\serial_benchmark_results_for_model_fit_2.csv")
size = df["size"].to_numpy(dtype=float)
avg = df["avg"].to_numpy()
std = df["std"].to_numpy()


# --- Model: time(size) = c * size * log(log(size)) ---
def model_x(size):
    return size * np.log(np.log(size))
 
x = model_x(size)
 
# Weighted linear regression through the origin (single parameter c),
# weighting each point by 1/std^2 (inverse-variance weighting):
# minimize sum( w * (avg - c*x)^2 )  =>  c = sum(w*x*avg) / sum(w*x^2)
w = 1.0 / std**2
c = np.sum(w * x * avg) / np.sum(w * x**2)
 
# Predicted values
avg_pred = c * x
 
# Weighted R^2 (consistent with the weighted least-squares objective):
# uses the weighted mean as the baseline for SS_tot
avg_mean_weighted = np.sum(w * avg) / np.sum(w)
ss_res = np.sum(w * (avg - avg_pred) ** 2)
ss_tot = np.sum(w * (avg - avg_mean_weighted) ** 2)
r2 = 1 - ss_res / ss_tot
 
# print(f"Fitted c       = {c:.6e}")
# print(f"R^2 (weighted) = {r2:.6f}")
 
# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))
 
# Data with error bars
ax.errorbar(size, avg, yerr=std, fmt='o', color='C0', ecolor='C0',
            elinewidth=1, capsize=3, markersize=5, label='Measured data')
 
# Smooth model curve
size_smooth = np.logspace(np.log10(size.min()), np.log10(size.max()), 200)
time_smooth = c * model_x(size_smooth)
ax.plot(size_smooth, time_smooth, '-', color='C1', linewidth=1.5,
         label=fr'Weighted fit: $t = c \cdot n \log(\log n)$, $c={c:.3e}$, $R^2={r2:.4f}$')
 
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('size (n)')
ax.set_ylabel('time (s)')
ax.set_title('Sieve of Eratosthenes: execution time vs. input size')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
 
fig.tight_layout()
fig.savefig('run_data/sieve_model_fit.png', dpi=150)
print("Plot saved.")