# ./measuring/plot_exec_time_comparison.py
"""
Plot execution time vs. input size (log-log) for several sieve variants,
each with error bars and its own weighted fit to t = c * n * log(log(n)).

Usage:
    python -m measuring.plot_exec_time_comparison 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- the theoretical model for Sieve of Eratosthenes: t = c * n * log(log(n)) ---
def model(n, c):
    return c * n * np.log(np.log(n))


INPUT_CSV = "run_data/numpy_vect_compare_2026-07-04T18-42-26.csv"
OUTPUT_PNG = "run_data/numpy_vect_compare.png"
 

# Display settings: one (color, label) per function name found in the data.
# Add more entries here if you plot additional variants later.
DISPLAY = {
    "serial_bitarray_sieve": {
        "label": "Serial (bitarray)",
        "color": "tab:blue",
        "marker": "o",
    },
    "serial_bool_sieve": {
        "label": "Serial (np.bool_)",
        "color": "tab:red",
        "marker": "s",
    },
       "numpy_vectorized_inner_loop": {
        "label": "Numpy vectorisation (inner loop)",
        "color": "tab:green",
        "marker": "^",
    },
    # "numba_parallel_inner_loop_bool_sieve": {
    #     "label": "Numba parallel (inner loop)",
    #     "color": "tab:green",
    #     "marker": "^",
    # },
}

def fit_and_plot(ax, sub_df, style):
    """Weighted fit + scatter-with-errorbars + fit line for one function's data."""
    n = sub_df["size"].to_numpy(dtype=float)
    t = sub_df["avg [s]"].to_numpy(dtype=float)
    std = sub_df["std [s]"].to_numpy(dtype=float)
 
    # avoid zero-weight blow-ups if any std happens to be 0
    sigma = np.where(std > 0, std, std[std > 0].min() if np.any(std > 0) else 1.0)
 
    popt, pcov = curve_fit(model, n, t, sigma=sigma, absolute_sigma=True, p0=[1e-8])
    c_fit = popt[0]
    c_err = np.sqrt(np.diag(pcov))[0]
 
    # R^2 (weighted residuals against weighted mean, simple unweighted version shown)
    residuals = t - model(n, c_fit)
    ss_res = np.sum((residuals / sigma) ** 2)
    ss_tot = np.sum(((t - np.average(t, weights=1 / sigma**2)) / sigma) ** 2)
    r2 = 1 - ss_res / ss_tot
 
    # measured points with error bars
    ax.errorbar(
        n, t, yerr=std,
        fmt=style["marker"], color=style["color"],
        markersize=6, capsize=3, linestyle="none",
        label=f'{style["label"]} (measured)',
    )
 
    # smooth fit line across the measured range
    n_fit = np.logspace(np.log10(n.min()), np.log10(n.max()), 200)
    ax.plot(
        n_fit, model(n_fit, c_fit),
        color=style["color"], linestyle="-", linewidth=1.5,
        label=f'{style["label"]} fit: c={c_fit:.3e} ($R^2$={r2:.4f})',
    )
 
    return c_fit, c_err, r2
 
 
def main():
    df = pd.read_csv(INPUT_CSV)
    print(df)
 
    fig, ax = plt.subplots(figsize=(10, 7))
 
    results = {}
    for func_name, sub_df in df.groupby("function"):
        style = DISPLAY.get(func_name, {
            "label": func_name, "color": None, "marker": "d"
        })
        sub_df = sub_df.sort_values("size")
        print(sub_df, style)
        results[func_name] = fit_and_plot(ax, sub_df, style)
 
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("size (n)")
    ax.set_ylabel("time (s)")
    ax.set_title("Sieve of Eratosthenes variants: execution time vs. input size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG) or ".", exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved plot to {OUTPUT_PNG}")
 
    print("\nFitted constants (t = c * n * log(log(n))):")
    for func_name, (c_fit, c_err, r2) in results.items():
        print(f"  {func_name:45s} c = {c_fit:.4e} +/- {c_err:.1e}   R^2 = {r2:.4f}")
 
 
if __name__ == "__main__":
    main()