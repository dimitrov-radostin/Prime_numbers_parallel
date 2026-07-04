# measuring/time_and_memory_meas_for_comparison.py
#
# run with python -m measuring.time_and_memory_meas_for_comparison
#

import subprocess
import json
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm_versions.serial_versions import (
    serial_int_array_sieve_no_return,
    serial_bitarray_sieve_no_return,
    serial_bool_sieve_no_return,
)

serial_versions = [
    serial_int_array_sieve_no_return,
    serial_bitarray_sieve_no_return,
    serial_bool_sieve_no_return,
]


def measure_peak_via_subprocess(module_name, func_name, N, python_exe=None):
    """
    Call the worker as a module so imports resolve correctly:
      python -m measuring.measure_peak_subprocess_psutil algorithm_versions.serial_versions func_name N

    Returns (runtime_s, rss_bytes).
    """
    if python_exe is None:
        python_exe = sys.executable

    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        python_exe,
        "-m",
        "measuring.measure_peak_subprocess_psutil",
        module_name,
        func_name,
        str(N),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(project_root),
        env=os.environ.copy(),
    )

    if proc.returncode != 0:
        # include stderr for debugging
        raise RuntimeError(
            f"Subprocess failed (rc={proc.returncode}).\ncmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    out = json.loads(proc.stdout)

    # Accept either "rss_bytes" or "peak_rss_bytes" or "final_rss_bytes" depending on worker
    rss = (
        out.get("rss_bytes") or out.get("peak_rss_bytes") or out.get("final_rss_bytes")
    )
    if rss is None:
        raise KeyError(
            "Worker output did not contain rss_bytes/peak_rss_bytes/final_rss_bytes"
        )

    return float(out["runtime_s"]), int(rss)


def main():
    N_stat = 5
    sizes = np.array([50_000_000, 100_000_000, 200_000_000, 300_000_000])

    run_date = datetime.datetime.now().isoformat(timespec="seconds")
    records = []

    for f in serial_versions:
        func_name = f.__name__
        for N in sizes:
            times = []
            memories = []
            for i in range(N_stat):
                t, m = measure_peak_via_subprocess(
                    "algorithm_versions.serial_versions", func_name, N
                )
                print(f"{func_name} N={N} run={i} time={t:.6f}s rss={m} bytes")
                times.append(t)
                memories.append(m)

            row = {
                "function": func_name,
                "size": int(N),
                "run_date": run_date,
                "time_avg": float(np.mean(times)),
                "time_std": float(np.std(times, ddof=1)),
                "mem_avg": float(np.mean(memories)),
                "mem_std": float(np.std(memories, ddof=1)),
            }
            # store individual reps if desired
            row.update({f"time_rep_{i}": float(v) for i, v in enumerate(times)})
            row.update({f"mem_rep_{i}": int(v) for i, v in enumerate(memories)})
            records.append(row)

    df = pd.DataFrame.from_records(records)
    # out_csv = Path(__file__).resolve().parent.parent / "run_data" / f"mem_time_results_{run_date}.csv"
    safe_run_date = run_date.replace(":", "-")  # 2026-07-02T19-29-52
    out_csv = (
        Path(__file__).resolve().parent.parent
        / "run_data"
        / f"mem_time_results_{safe_run_date}.csv"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Results written to {out_csv}")
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
