##
# run with python -m measuring.time_parallel_funcs_1
##

import os
import time
import sys
import datetime
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess

# --- Configure these to match your implementation module/function ---
# The module must be importable from the project root (use python -m or PYTHONPATH=. when running)
MODULE_NAME = "algorithm_versions.parallel_versions"          # e.g., file sieve_numba_fixed.py
FUNC_NAME = "numba_parallel_inner_loop_bool_sieve"       # function compiled with numba
# ------------------------------------------------------------------

# Thread sweep and sizes to test
THREADS_TO_TEST = [1, 2, 4, 6, 8, 10, 12, 14, 16]
SIZES = [10_000_000, 100_000_000, 500_000_000, 1_000_000_000]
N_stat = 3                                # repetitions per (N,threads)

# CSV output
run_date = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
out_dir = Path.cwd() / "run_data"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / f"numba_thread_sweep_{run_date}.csv"




# Child runner: runs in a fresh Python process and prints JSON with runtime
def time_single_run_subprocess(module_name, func_name, N, n_threads, warmup=True, timeout=None):
    """
    Runs the target function in a fresh Python process with n_threads.
    Returns runtime in seconds (float).
    """
    # Build a small Python snippet to run in the child process.
    # It sets env vars, sets numba threads, imports module and runs the function once,
    # optionally does a warmup call (to compile) and then a timed call.
    py = rf"""
import os, time, json, importlib, gc
# set env vars early in the child
os.environ['NUMBA_NUM_THREADS'] = '{n_threads}'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# import numba and set runtime thread count before any parallel region
import numba
numba.set_num_threads({n_threads})

# import target module and function
mod = importlib.import_module('{module_name}')
func = getattr(mod, '{func_name}')

# optional warmup to compile (small N)
if {1 if warmup else 0}:
    gc.collect()
    try:
        func(1000)
    except Exception:
        # ignore warmup errors; we still attempt the timed run
        pass

# timed run
gc.collect()
t0 = time.perf_counter()
func({N})
t1 = time.perf_counter()
print(json.dumps({{'runtime_s': t1 - t0}}))
"""
    proc = subprocess.run([sys.executable, "-c", py],
                          capture_output=True, text=True, check=False, timeout=timeout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Child process failed (rc={proc.returncode}).\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )

    # parse last non-empty line as JSON
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not out_lines:
        raise RuntimeError(f"No output from child. stderr:\n{proc.stderr}")
    out = json.loads(out_lines[-1])
    return float(out["runtime_s"])

def main():
    records = []
    baseline_times = {}

    for N in SIZES:
        # baseline: threads == 1
        print(f"\nMeasuring baseline (1 thread) for N={N}")
        times = []
        for rep in range(N_stat):
            t = time_single_run_subprocess(MODULE_NAME, FUNC_NAME, N, n_threads=1, warmup=True)
            print(f"  baseline rep {rep}: time={t:.6f}s")
            times.append(t)
        baseline_avg = float(np.mean(times))
        baseline_std = float(np.std(times, ddof=1))
        baseline_times[N] = baseline_avg

        records.append({
            "function": FUNC_NAME,
            "N": int(N),
            "threads": 1,
            "run_date": run_date,
            "time_avg": baseline_avg,
            "time_std": baseline_std,
            "speedup_avg": 1.0,
            **{f"time_rep_{i}": float(v) for i, v in enumerate(times)}
        })

        # sweep other thread counts
        for threads in THREADS_TO_TEST:
            if threads == 1:
                continue
            print(f"\nMeasuring threads={threads} for N={N}")
            times = []
            for rep in range(N_stat):
                t = time_single_run_subprocess(MODULE_NAME, FUNC_NAME, N, n_threads=threads, warmup=True)
                print(f"  threads={threads} rep {rep}: time={t:.6f}s")
                times.append(t)
            avg = float(np.mean(times))
            std = float(np.std(times, ddof=1))
            speedup = baseline_times[N] / avg if avg > 0 else float("nan")

            records.append({
                "function": FUNC_NAME,
                "N": int(N),
                "threads": int(threads),
                "run_date": run_date,
                "time_avg": avg,
                "time_std": std,
                "speedup_avg": float(speedup),
                "baseline_time": baseline_times[N],
                **{f"time_rep_{i}": float(v) for i, v in enumerate(times)}
            })

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"\nResults written to {out_csv}")

if __name__ == "__main__":
    # run from project root so child can import MODULE_NAME
    main()