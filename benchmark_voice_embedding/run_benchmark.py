"""
Vergleicht die in der Konstanten MODELS gelisteten Sprecher-Embedding-Modelle auf den VoxCeleb1-Verifikations-Trials.
Für jedes Modell zuerst Modell laden und anschließend für jeden Trial (Audiopaar) Embeddings berechnen und
dabei Laufzeit, RAM- und CPU-Verbrauch messen (measure_specs).
Cosine Similarity der Embeddings wird als List gesammelt und anschließend werden die
Erkennungsmetriken (EER, minDCF, TAR@FAR) berechnet.
Alle Ergebnisse der Modelle landen gesammelt in results/summary.json.
"""
import gc
import os
import json
import psutil
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm # Fortschrittsbalken in der Konsole
from benchmark.metrics import all_metrics
from benchmark.performance import measure_specs
from benchmark.load_dataset import load_voxceleb1_veri_test
from benchmark.frameworks.nemo_model import NeMo
from benchmark.frameworks.pyannote_model import Pyannote
from benchmark.frameworks.speechbrain_model import SpeechBrain
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

MODELS = [
    # SpeechBrain("speechbrain/spkrec-xvect-voxceleb", device="cuda"), # k2-Abhängigkeit, kein Windows-Support
    Pyannote("pyannote/embedding", device="cuda"),
    Pyannote("pyannote/wespeaker-voxceleb-resnet34-LM", device="cuda"),
    SpeechBrain("speechbrain/spkrec-ecapa-voxceleb", device="cuda"),
    SpeechBrain("speechbrain/spkrec-resnet-voxceleb", device="cuda"),
    NeMo("titanet_large", device="cuda"),
    NeMo("titanet_small", device="cuda"),
    NeMo("speakerverification_speakernet", device="cuda"),
    NeMo("ecapa_tdnn", device="cuda"),
]

trials = load_voxceleb1_veri_test()
results = []

proc = psutil.Process()

for model in MODELS:
    name = model.model_name
    print(f"{name}")
    embedding_stats = [] # Liste von Dikt-Objekten mit "zeit_s", "ram_avg_mb", "ram_max_mb", "cpu_%"
    labels = [] # 1 = gleicher Sprecher, 0 = verschiedene Sprecher
    scores = [] # Cosine Similarity der Embeddings, je höher desto ähnlicher

    # Vorheriges Modell aus dem Speicher freigeben bevor RAM gemessen wird
    gc.collect() # Python-Garbage-Collector erzwingen: gibt Objekte frei, deren Referenzzähler 0 ist (z.B. das vorherige Modell)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ram_t0 = proc.memory_info().rss / 1024 / 1024
    model.load()
    idle_ram_mb = proc.memory_info().rss / 1024 / 1024 - ram_t0

    for trial in tqdm(trials, desc="Trials"):
        emb1, stats1 = measure_specs(model.embedding, trial["audio1"])
        emb2, stats2 = measure_specs(model.embedding, trial["audio2"])
        embedding_stats.append(stats1)
        embedding_stats.append(stats2)
        labels.append(trial["label"])
        scores.append(float(cosine_similarity([emb1], [emb2])[0][0]))

    if cuda_available:
        vram_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        vram_peak_mb = 0

    metrics = all_metrics(labels, scores)
    results.append({
        "modell": name,
        "idle_ram_mb": round(idle_ram_mb, 2),
        "vram_peak_mb": round(vram_peak_mb, 2),
        "inf_mean_s": round(np.mean([t["zeit_s"] for t in embedding_stats]), 3),
        "ram_max_mb": round(max(x["ram_max_mb"] for x in embedding_stats), 2),
        "ram_avg_mb": round(np.mean([x["ram_avg_mb"] for x in embedding_stats]), 2),
        "cpu_%": round(np.mean([x["cpu_%"] for x in embedding_stats]), 2),
        **metrics,
    })

os.makedirs("results", exist_ok=True) # Ordner für Ergebnisse erstellen, falls nicht vorhanden

with open("results/summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

