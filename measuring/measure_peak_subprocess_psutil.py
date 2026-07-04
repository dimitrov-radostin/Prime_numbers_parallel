# measuring/measure_peak_subprocess_psutil.py
# pip install psutil

import json
import time
import sys
import importlib
import gc
import psutil
import threading


def run_and_sample_memory(module_name, func_name, N, interval=0.01):
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    proc = psutil.Process()
    samples = []
    stop = False

    def sampler():
        while not stop:
            samples.append(proc.memory_info().rss)
            time.sleep(interval)

    gc.collect()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    # run but do not return the big object; let the function create and discard internally
    func(N)
    t1 = time.time()
    stop = True
    th.join()
    peak = max(samples) if samples else proc.memory_info().rss
    avg = sum(samples)/len(samples) if samples else peak
    print(json.dumps({"runtime_s": t1-t0, "peak_rss_bytes": peak, "avg_rss_bytes": avg}))


if __name__ == "__main__":
    module_name, func_name, N_max = sys.argv[1], sys.argv[2], int(sys.argv[3])
    run_and_sample_memory(module_name, func_name, N_max)


# def run_and_report(module_name, func_name, N_max):
#     mod = importlib.import_module(module_name)
#     func = getattr(mod, func_name)

#     # warmup / reduce noise
#     gc.collect()
#     proc = psutil.Process()

#     # sample RSS before run (baseline)
#     baseline = proc.memory_info().rss

#     t0 = time.time()
#     result = func(N_max)
#     t1 = time.time()

#     # sample RSS after run and get peak if available
#     # psutil does not expose peak RSS cross-platform reliably,
#     # but on Linux you can read /proc/<pid>/status or use psutil.memory_info().rss samples.
#     # Here we return final RSS and runtime; caller can sample if it needs peak timeline.
#     final_rss = proc.memory_info().rss

#     out = {
#         "final_rss_bytes": int(final_rss),
#         "baseline_rss_bytes": int(baseline),
#         "rss_bytes": int(final_rss) - int(baseline),
#         "runtime_s": float(t1 - t0)
#     }
#     sys.stdout.write(json.dumps(out))

# measuring/measure_peak_subprocess_psutil.py