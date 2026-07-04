import numpy as np
import timeit 
import functools
import datetime
import pandas as pd


from serial_versions import serial_int_array_sieve, serial_bitarray_sieve, serial_bool_sieve
serial_versions = [serial_int_array_sieve, serial_bitarray_sieve, serial_bool_sieve]

N_stat = 10
sizes = np.array([10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000])

run_date = datetime.datetime.now().isoformat(timespec="seconds")
records = []

for f in serial_versions:
    func_name = f.__name__
    for N in sizes:
        wrapped = functools.partial(f, N_max=N)
        t = timeit.repeat(wrapped, number=1, repeat=N_stat)

        row = {
            "function": func_name,
            "size": N,
            "run_date": run_date,
            "avg": np.mean(t),
            "std": np.std(t, ddof=1),
        }
        row.update({f"rep_{i}": val for i, val in enumerate(t)})
        records.append(row)

df = pd.DataFrame.from_records(records)

# Explicit column order: metadata + avg/std before the rep_* measurements
rep_cols = [f"rep_{i}" for i in range(N_stat)]
df = df[["function", "size", "run_date", "avg", "std"] + rep_cols]

df.to_csv("../run_data/serial_benchmark_results_3.csv", index=False)

