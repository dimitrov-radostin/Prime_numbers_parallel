##
# run with python -m measuring.time_parallel_funcs_1
##

import os
import time
import datetime
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import shlex

# --- Configure these to match your implementation module/function ---
# The module must be importable from the project root (use python -m or PYTHONPATH=. when running)
MODULE_NAME = "algorithm_versions.parallel_versions"          # e.g., file sieve_numba_fixed.py
FUNC_NAME = "numba_parallel_inner_loop_bool_sieve"       # function compiled with numba
# ------------------------------------------------------------------

# Thread sweep and sizes to test
THREADS_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8]            # include 1 for baseline
SIZES = [10_000_000, 100_000_000, 500_000_000, 1_000_000_000]
N_stat = 3                                # repetitions per (N,threads)

# CSV output
run_date = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
out_dir = Path.cwd() / "run_data"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / f"numba_thread_sweep_{run_date}.csv"

# Import target function (do this after any env var settings if needed)
import importlib
mod = importlib.import_module(MODULE_NAME)
func = getattr(mod, FUNC_NAME)

# Import numba to control threads
import numba

def set_thread_env(n_threads):
    """
    Set environment variables that affect threading. Must be done before importing
    heavy libraries if you want them to pick up the values at import time.
    We still call numba.set_num_threads at runtime below.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # NUMBA_NUM_THREADS can be set before import; we set at runtime too
    os.environ["NUMBA_NUM_THREADS"] = str(n_threads)

def time_single_run_subprocess(module_name, func_name, N, n_threads, timeout=None):
    """
    Run: python - <<'PY' 
             import os, numba, importlib, time, json
             os.environ['NUMBA_NUM_THREADS'] = 'n_threads'
             numba.set_num_threads(n_threads)
             mod = importlib.import_module(module_name)
             func = getattr(mod, func_name)
             t0=time.perf_counter(); func(N); t1=time.perf_counter()
             print(json.dumps({'runtime_s': t1-t0}))
           PY
    Returns runtime_s (float).
    """
    # Build a small Python snippet to run in the child process.
    # It sets env var, sets numba threads, imports module and runs the function once,
    # then prints a single JSON line with runtime.
    py = rf"""
import os, time, json, importlib, numba, gc
# set env var early (child process inherits env but we set explicitly)
os.environ['NUMBA_NUM_THREADS'] = '{n_threads}'
# also limit other thread pools to avoid oversubscription
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# set runtime thread count before any parallel region
numba.set_num_threads({n_threads})

# import target and run
mod = importlib.import_module('{module_name}')
func = getattr(mod, '{func_name}')
gc.collect()
t0 = time.perf_counter()
func({N})
t1 = time.perf_counter()
print(json.dumps({{'runtime_s': t1 - t0}}))
"""

    # Run the snippet with the same Python interpreter
    proc = subprocess.run([sys.executable, "-c", py],
                          capture_output=True, text=True, check=False, timeout=timeout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (rc={proc.returncode}).\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )

    # parse JSON printed by child
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return float(out["runtime_s"])

def warmup_compile(N=1000):
    """One small call to trigger Numba compilation before timing."""
    # ensure a consistent thread setting for compilation
    numba.set_num_threads(1)
    func(N)

def main():
    # Warmup / compile
    print("Warming up / compiling Numba function...")
    warmup_compile(2000)

    records = []
    # We'll compute baseline per N (threads==1)
    baseline_times = {}

    # for N in SIZES:
    #     # baseline (1 thread) first
    #     print(f"\nMeasuring baseline (1 thread) for N={N}")
    #     times = []
    #     for rep in range(N_stat):
    #         t, actual_threads = time_single_run(1, N)
    #         print(f"  baseline rep {rep}: time={t:.6f}s threads={actual_threads}")
    #         times.append(t)
    #     baseline_avg = float(np.mean(times))
    #     baseline_std = float(np.std(times, ddof=1))
    #     baseline_times[N] = baseline_avg

    #     # store baseline row
    #     records.append({
    #         "function": FUNC_NAME,
    #         "N": int(N),
    #         "threads": 1,
    #         "run_date": run_date,
    #         "time_avg": baseline_avg,
    #         "time_std": baseline_std,
    #         "speedup_avg": 1.0,               # baseline speedup = 1
    #         "baseline_time": baseline_avg,
    #         **{f"time_rep_{i}": float(v) for i, v in enumerate(times)}
    #     })

    #     # now sweep other thread counts (including 1 again if desired)
    #     for threads in THREADS_TO_TEST:
    #         if threads == 1:
    #             # already measured baseline; skip or keep duplicate
    #             continue
    #         print(f"\nMeasuring threads={threads} for N={N}")
    #         times = []
    #         for rep in range(N_stat):
    #             t, actual_threads = time_single_run(threads, N)
    #             print(f"  threads={actual_threads} rep {rep}: time={t:.6f}s")
    #             times.append(t)
    #         avg = float(np.mean(times))
    #         std = float(np.std(times, ddof=1))
    #         speedup = baseline_times[N] / avg if avg > 0 else float("nan")

    #         records.append({
    #             "function": FUNC_NAME,
    #             "N": int(N),
    #             "threads": int(threads),
    #             "run_date": run_date,
    #             "time_avg": avg,
    #             "time_std": std,
    #             "speedup_avg": float(speedup),
    #             "baseline_time": baseline_times[N],
    #             **{f"time_rep_{i}": float(v) for i, v in enumerate(times)}
    #         })
    #         print(threads, avg)
    
    for N in SIZES:
        baseline_time = time_single_run_subprocess(MODULE_NAME, FUNC_NAME, N, n_threads=1)
        
        for threads in THREADS_TO_TEST:
            t = time_single_run_subprocess(MODULE_NAME, FUNC_NAME, N, n_threads=threads)
            speedup = baseline_time / t
            print(N, threads, speedup)

    # Save CSV
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"\nResults written to {out_csv}")

if __name__ == "__main__":
    # Optionally set env vars before heavy imports (done earlier in script)
    set_thread_env(max(THREADS_TO_TEST))
    main()
