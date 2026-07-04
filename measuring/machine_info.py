import platform
import os
import numpy as np
import subprocess
import json


def get_cache_info_windows_wmic():
    cache_info = {}
    try:
        # L1, L2, L3 cache size in KB
        result = subprocess.run(
            ["wmic", "cpu", "get", "L2CacheSize,L3CacheSize", "/format:list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if "=" in line and line.split("=")[1]:
                key, val = line.split("=")
                cache_info[key] = f"{val}K"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return cache_info


def get_cache_info_windows_powershell():
    cache_info = {}
    try:
        cmd = (
            "Get-CimInstance Win32_CacheMemory | "
            "Select-Object Level,InstalledSize | ConvertTo-Json"
        )
        result = subprocess.run(
            ["powershell", "-Command", cmd], capture_output=True, text=True, timeout=5
        )
        import json

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
        for entry in data:
            level = entry.get("Level")
            size_kb = entry.get("InstalledSize")
            cache_info[f"L{level}_cache"] = f"{size_kb}K"
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return cache_info


def get_cache_info_windows():
    info = get_cache_info_windows_powershell()
    if not info:
        info = get_cache_info_windows_wmic()
    return info


def get_cache_info_linux():
    cache_info = {}
    cache_base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        for entry in sorted(os.listdir(cache_base)):
            index_path = os.path.join(cache_base, entry)
            if not entry.startswith("index"):
                continue
            try:
                with open(os.path.join(index_path, "level")) as f:
                    level = f.read().strip()
                with open(os.path.join(index_path, "type")) as f:
                    cache_type = f.read().strip()
                with open(os.path.join(index_path, "size")) as f:
                    size = f.read().strip()
                key = f"L{level}_{cache_type}"
                cache_info[key] = size
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        pass
    return cache_info


def get_cache_info_cpuinfo():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        return {
            "l1_data_cache_size": info.get("l1_data_cache_size"),
            "l1_instruction_cache_size": info.get("l1_instruction_cache_size"),
            "l2_cache_size": info.get("l2_cache_size"),
            "l3_cache_size": info.get("l3_cache_size"),
        }
    except ImportError:
        return {}


def get_machine_info(name: str = "", save: bool = False):
    info = {
        "ref_name": name,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
    }

    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        info["cpu_count_physical"] = None
        info["ram_gb"] = None

    if platform.system() == "Windows":
        info.update(get_cache_info_windows())
    elif platform.system() == "Linux":
        info.update(get_cache_info_linux())

    if save and name != "":
        with open(f"run_data/{name}_machine_info.json", "w") as f:
            json.dump(info, f, indent=2)

    return info


# print(get_machine_info('test', True))

