# measuring/time_meas_for_comparison.py
#
# run with python -m measuring.time_meas_for_comparison
#

import numpy as np
import timeit
import functools
import datetime
import pandas as pd
from pathlib import Path

from algorithm_versions.serial_versions import (
    serial_int_array_sieve,
    serial_bitarray_sieve,
    serial_bool_sieve,
)
from algorithm_versions.inner_loop_opt_versions import (
    numba_parallel_inner_loop_bool_sieve,
    numpy_vectorized_inner_loop,
)

# versions_for_compare = [serial_int_array_sieve, serial_bitarray_sieve, serial_bool_sieve]
versions_for_compare = [
    serial_bitarray_sieve,
    serial_bool_sieve,
    numpy_vectorized_inner_loop,
]

machine_ref_name = "PC_AMD_Ryzen_7_7735HS"

N_stat = 10
sizes = np.array(
    [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]) #, 500_000_000])

run_date = datetime.datetime.now().isoformat(timespec="seconds")
records = []

for f in versions_for_compare:
    func_name = f.__name__
    for N in sizes:
        wrapped = functools.partial(f, N_max=N)
        t = timeit.repeat(wrapped, number=1, repeat=N_stat)

        row = {
            "machine": machine_ref_name,
            "function": func_name,
            "size": N,
            "run_date": run_date,
            "avg [s]": np.mean(t),
            "std [s]": np.std(t, ddof=1),
            "rel_variance [%]": 100 * np.std(t, ddof=1) / np.mean(t),
        }
        row.update({f"rep_{i}": val for i, val in enumerate(t)})
        records.append(row)
        print(func_name, N, np.mean(t))

df = pd.DataFrame.from_records(records)

# Explicit column order: metadata + avg/std before the rep_* measurements
# rep_cols = [f"rep_{i}" for i in range(N_stat)]
# df = df[["function", "size", "run_date", "avg", "std"] + rep_cols]

# df.to_csv("../run_data/serial_benchmark_results_3.csv", index=False)

df = pd.DataFrame.from_records(records)
# out_csv = Path(__file__).resolve().parent.parent / "run_data" / f"mem_time_results_{run_date}.csv"
safe_run_date = run_date.replace(":", "-")  # 2026-07-02T19-29-52
out_csv = (
    Path(__file__).resolve().parent.parent
    / "run_data"
    / f"numpy_vect_compare_{safe_run_date}.csv"
)

out_csv.parent.mkdir(parents=True, exist_ok=True)
print(f"Results written to {out_csv}")
df.to_csv(out_csv, index=False)
