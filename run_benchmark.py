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
from benchmark.frameworks.nemo_adapter import NeMo
# from benchmark.frameworks.speechbrain_adapter import SpeechBrain
from benchmark.frameworks.pyannote_adapter import Pyannote
# from benchmark.frameworks.picovoice_adapter import Picovoice
from sklearn.metrics.pairwise import cosine_similarity
from benchmark.visualize import all_plots_generate

load_dotenv()

MODELS = [
    # NeMo("titanet_large", device="cuda"),
    NeMo("titanet_small", device="cuda"),
    # NeMo("speakerverification_speakernet", device="cuda"),
    # NeMo("ecapa_tdnn", device="cuda"),
    # SpeechBrain("speechbrain/spkrec-ecapa-voxceleb", device="cuda"),
    # SpeechBrain("speechbrain/spkrec-xvect-voxceleb", device="cuda"),
    Pyannote("pyannote/embedding", device="cuda"),
    # Picovoice(device="cuda"),
]

trials = load_voxceleb1_veri_test()
results = []

for model in MODELS:
    name = f"{model.__class__.__name__}"
    print(f"{name}")
    embedding_stats = [] # Liste von Dikt-Objekten mit "zeit_s", "ram_avg_mb", "ram_max_mb", "cpu_%"
    labels = [] # 1 = gleicher Sprecher, 0 = verschiedene Sprecher
    scores = [] # Cosine Similarity der Embeddings, je höher desto ähnlicher

    # RAM und VRAM vor und nach Laden des Modells messen
    proc = psutil.Process()
    ram_t0 = proc.memory_info().rss / 1024 / 1024
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats() # VRAM-Peak nach Laden des Modells messen, deshalb hier zurücksetzen
    model.load()
    idle_ram_mb  = proc.memory_info().rss / 1024 / 1024 - ram_t0
    if cuda_available:
        idle_vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        idle_vram_mb = 0

    for trial in tqdm(trials[:100], desc="Trials"): # erstmal nur 100 zum testen
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
        "idle_vram_mb": round(idle_vram_mb, 2),
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

all_plots_generate("results")
