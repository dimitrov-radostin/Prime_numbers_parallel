"""
Benchmark: serial baseline vs. serial_segmented(N, d) vs. parallel_segmented(N, d, k).

Usage:
    python -m measuring.time_meas_parallel_segmented

Notes:
- Timing brackets ONLY the function call (time.perf_counter()), not any
  interpreter/import overhead -- consistent with the earlier serial/Numba benchmarks.
- parallel_segmented creates a fresh Pool per call (see parallel_segmented.py),
  so this measures "cold" cost, including per-call worker spawn overhead.
  That overhead is expected to be non-trivial on Windows (spawn, not fork).
  If you also want "warm" (pool reused across reps) numbers, say so and
  we can add a second code path for that -- it requires refactoring
  parallel_segmented to accept an externally-supplied Pool.
- Must be run as a script (the __main__ guard below is required on Windows,
  since multiprocessing re-imports this module in every spawned worker).
"""
import csv
import os
import time
import datetime
import math
import numpy as np
from multiprocessing import Pool

from algorithm_versions.inner_loop_opt_versions import numpy_vectorized_inner_loop  
from algorithm_versions.segmented_versions import serial_segmented
from algorithm_versions.multiprocessing_segmented import parallel_segmented_with_pool, serial_sieve, _init_worker

MACHINE_NAME = "PC_AMD_Ryzen_7_7735HS"  
REPS = 5
OUTPUT_CSV = "run_data/parallel_segmented_benchmark_updated.csv"

N_VALUES = [10_000_000, 100_000_000, 200_000_000]
D_VALUES = [100_000, 250_000, 400_000, 500_000]   # centered on the L2-boundary region found earlier
K_VALUES = [1, 2, 4, 6, 8]


 
def timed_run(func, *args, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = float(np.mean(times))
    std = float(np.std(times))
    rel_var = 100.0 * std / avg if avg > 0 else float("nan")
    return avg, std, rel_var, result
 
 
def write_row(writer, function_name, N, d, k, avg, std, rel_var):
    writer.writerow({
        "machine": MACHINE_NAME,
        "function": function_name,
        "N": N,
        "d": d if d is not None else "",
        "k": k if k is not None else "",
        "run_date": datetime.datetime.now().isoformat(timespec="seconds"),
        "avg_s": avg,
        "std_s": std,
        "rel_variance_pct": rel_var,
    })
 
 
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["machine", "function", "N", "d", "k", "run_date", "avg_s", "std_s", "rel_variance_pct"]
 
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
 
        for N in N_VALUES:
            # --- baseline: plain serial (no d, no k) ---
            avg, std, rel_var, ref_primes = timed_run(numpy_vectorized_inner_loop, N)
            write_row(writer, "numpy_vectorized_inner_loop", N, None, None, avg, std, rel_var)
            print(f"N={N:>12,d}  serial baseline           avg={avg:.4f}s std={std:.4f}")
            f.flush()
 
            # --- serial_segmented(N, d): sweep d, k fixed at 1 (no parallelism) ---
            for d in D_VALUES:
                avg, std, rel_var, primes = timed_run(serial_segmented, N, d)
                assert np.array_equal(primes, ref_primes), f"correctness check failed: N={N}, d={d}"
                write_row(writer, "serial_segmented", N, d, 1, avg, std, rel_var)
                print(f"N={N:>12,d}  serial_segmented d={d:>8,d}  avg={avg:.4f}s std={std:.4f}")
                f.flush()
 
            # --- parallel_segmented: measure pool startup cost and warm per-call cost separately ---
            for k in K_VALUES:
                N_sqrt = math.isqrt(N - 1)
                seed = serial_sieve(N_sqrt + 1)
 
                # 1) pool startup cost alone (create + immediately tear down, no work)
                startup_times = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    with Pool(processes=k, initializer=_init_worker, initargs=(seed,)) as pool:
                        pass
                    t1 = time.perf_counter()
                    startup_times.append(t1 - t0)
                avg = float(np.mean(startup_times))
                std = float(np.std(startup_times))
                rel_var = 100.0 * std / avg if avg > 0 else float("nan")
                write_row(writer, "parallel_segmented_pool_startup", N, None, k, avg, std, rel_var)
                print(f"N={N:>12,d}  pool_startup k={k}  avg={avg:.4f}s std={std:.4f}")
                f.flush()
 
                # 2) warm per-call cost: one pool reused across the whole d-sweep for this (N, k)
                with Pool(processes=k, initializer=_init_worker, initargs=(seed,)) as pool:
                    for d in D_VALUES:
                        avg, std, rel_var, primes = timed_run(parallel_segmented_with_pool, N, d, pool)
                        assert np.array_equal(primes, ref_primes), f"correctness check failed: N={N}, d={d}, k={k}"
                        write_row(writer, "parallel_segmented_warm", N, d, k, avg, std, rel_var)
                        print(f"N={N:>12,d}  parallel_segmented_warm d={d:>8,d} k={k}  avg={avg:.4f}s std={std:.4f}")
                        f.flush()
 
    print(f"\nDone. Results written to {OUTPUT_CSV}")
 
 
if __name__ == "__main__":
    main()
 