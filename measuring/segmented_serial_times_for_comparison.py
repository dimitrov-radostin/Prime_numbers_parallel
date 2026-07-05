# measuring/segmented_serial_times_for_comparison.py
#
# run with python -m measuring.segmented_serial_times_for_comparison
#

import numpy as np
import timeit
import functools
import datetime
import pandas as pd
from pathlib import Path
import numba

from algorithm_versions.inner_loop_opt_versions import (
    numpy_vectorized_inner_loop    
)
from algorithm_versions.segmented_versions import (
    serial_segmented    
)

# versions_for_compare = [serial_int_array_sieve, serial_bitarray_sieve, serial_bool_sieve]
versions_for_compare = [
    # serial_bool_sieve,
    numpy_vectorized_inner_loop,
    serial_segmented
]

machine_ref_name = "PC_AMD_Ryzen_7_7735HS"

N_stat = 5
sizes = np.array(
    [10_000_000, 50_000_000, 100_000_000, 200_000_000]
    ) 
segment_sizes = np.array(
    [25_000, 30_000, 35_000, 400_000, 500_000, 600_000]
    ) 

# sizes = np.arange(5_000_000, 25_000_000, 1_000_000)

run_date = datetime.datetime.now().isoformat(timespec="seconds")
records = []
for f in versions_for_compare: 
    func_name = f.__name__   
        
    for N in sizes:
        if (func_name == 'serial_segmented'):
            for d in segment_sizes: 
                wrapped = functools.partial(f, N_max=N, d=d)
                t = timeit.repeat(wrapped, number=1, repeat=N_stat)

                row = {
                    "machine": machine_ref_name,
                    "function": func_name,
                    "segment size [B]": d,
                    "size": N,
                    "run_date": run_date,
                    "avg [s]": np.mean(t),
                    "std [s]": np.std(t, ddof=1),
                    "rel_variance [%]": 100 * np.std(t, ddof=1) / np.mean(t),
                }
                row.update({f"rep_{i}": val for i, val in enumerate(t)})
                records.append(row)
                print(func_name, N, d, np.mean(t))
        else:
            wrapped = functools.partial(f, N_max=N)
            t = timeit.repeat(wrapped, number=1, repeat=N_stat)
            row = {
                "machine": machine_ref_name,
                "function": func_name,
                "segment size [B]": '-',
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

df = pd.DataFrame.from_records(records)
# out_csv = Path(__file__).resolve().parent.parent / "run_data" / f"mem_time_results_{run_date}.csv"
safe_run_date = run_date.replace(":", "-")  # 2026-07-02T19-29-52
out_csv = (
    Path(__file__).resolve().parent.parent
    / "run_data"
    / f"serial_segmented_d_sweep_{safe_run_date}.csv"
)

out_csv.parent.mkdir(parents=True, exist_ok=True)
print(f"Results written to {out_csv}")
df.to_csv(out_csv, index=False)
