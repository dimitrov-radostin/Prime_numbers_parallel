import numpy as np
import timeit
import statistics
import functools
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from serial_versions import serial_bitarray_sieve

machine_ref_name = "PC_AMD_Ryzen_7_7735HS"  # fill-in manually after changing setup and after running the  get_machine_info() function

N_stat = 5
sizes = np.array(
    [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]
    )

run_date = datetime.datetime.now().isoformat(timespec="seconds")
records = []

for f in [serial_bitarray_sieve]:
    func_name = f.__name__
    for N in sizes:
        wrapped = functools.partial(f, N_max=N)
        t = timeit.repeat(wrapped, number=1, repeat=N_stat)

        row = {
            "machine": machine_ref_name,
            "run_date": run_date,
            "function": func_name,
            "size": N,
            "avg": np.mean(t),
            "std": np.std(t, ddof=1),
        }
        row.update({f"rep_{i}": val for i, val in enumerate(t)})
        records.append(row)

df = pd.DataFrame.from_records(records)

# Explicit column order: metadata + avg/std before the rep_* measurements
rep_cols = [f"rep_{i}" for i in range(N_stat)]
df = df[["machine_ref_name", "run_date", "function", "size", "avg", "std"] + rep_cols]

df.to_csv("../run_data/serial_benchmark_results_for_model_fit_{run_date}.csv", index=False)
