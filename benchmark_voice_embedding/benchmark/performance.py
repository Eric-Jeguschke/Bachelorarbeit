import time
import threading
import psutil
import numpy as np
from typing import Any, Callable

proc = psutil.Process() # Einmal erstellen, nicht pro Aufruf

def measure_specs(fn: Callable, *args: Any) -> tuple[Any, dict[str, float]]:
    """Misst Laufzeit, RAM- und CPU-Verbrauch während der Ausführung von fn(*args). Hier ist fn die Embedding-Funktion eines Modells, args sind die Eingabedaten (z.B. Pfad zu einer WAV-Datei)."""
    proc.cpu_percent() # CPU-Delta-Zähler zurücksetzen
    samples = []
    sampling_done = threading.Event()

    def sample():
        # RAM alle 100 ms samplen, bis stop gesetzt wird
        while not sampling_done.wait(0.1):
            samples.append(proc.memory_info().rss / 1024 / 1024)

    threading.Thread(target=sample, daemon=True).start()
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - t0
    sampling_done.set()

    if not samples: # Fallback falls Funktion schneller als 100 ms war
        samples.append(proc.memory_info().rss / 1024 / 1024)

    return result, {
        "zeit_s": round(elapsed, 3),
        "ram_max_mb": round(max(samples), 2),
        "ram_avg_mb": round(float(np.mean(samples)), 2),
        "cpu_%": round(proc.cpu_percent() / psutil.cpu_count(), 2), # normalisiert auf 0–100 % (psutil liefert Summe über alle Kerne)
    }
