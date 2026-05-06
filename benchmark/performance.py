import time
import psutil
import numpy as np
from memory_profiler import memory_usage

def measure_specs(function, *args): # args sind die Parameter für die Funktion
    proc = psutil.Process()
    proc.cpu_percent()  # Misst CPU-Auslastung seit letztem Aufruf, initial 0.0
    t0 = time.perf_counter()
    ram_samples, result = memory_usage((function, args), interval=0.1, retval=True)
    t1 = time.perf_counter() - t0
    return result, {
        "zeit_s":      round(t1, 3),
        "ram_peak_mb": round(max(ram_samples), 2),
        "ram_mean_mb": round(np.mean(ram_samples), 2),
        "cpu_%":       round(proc.cpu_percent(), 2),
    }