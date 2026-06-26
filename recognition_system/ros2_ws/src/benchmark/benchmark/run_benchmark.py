import soundfile as sf
import os
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

from microphone.voice_embedder import VoiceEmbedder
from benchmark.load_dataset import load_speakers
from benchmark.metrics import compute_confusion, find_best_threshold
from db.guest_store import find_by_voice, create_guest, clear_db

os.environ["DB_PATH"] = "/tmp/benchmark_db"  # Gilt nur für diesen Prozess, wird danach automatisch gelöscht


def enroll(enrolled_embeddings: dict):
    for speaker_id, embs in enrolled_embeddings.items():
        create_guest(
            name=str(speaker_id),
            voice_emb=embs[0].tolist(), # nur erstes Embedding pro Sprecher in DB, Rest sind für Wiedererkennungstest
            episode_json="{}",
            memory_json="{}",
            text_embedder=lambda text: [0.0] * 384,
        )


def collect_distances(enrolled_embeddings: dict, impostor_embeddings: dict) -> tuple[list, list]:
    distances = []
    labels = []

    # Echte Sprecher (Label 1): embs[1:] sind weitere Aufnahmen desselben Sprechers.
    for embs in enrolled_embeddings.values():
        for emb in embs[1:]:
            result = find_by_voice(emb.tolist(), threshold=2.0)  # 0 sind identische Vektoren, 1.0 sind orthogonale Vektoren und 2.0 sind entgegengesetzte Vektoren. Deswegen gibt immer Ergebnis zurück.
            if result is not None:
                distances.append(float(result["_distance"]))
                labels.append(1)

    # Fremde (Label 0): embs[0] ist nicht in der DB enrollt.
    for embs in impostor_embeddings.values():
        for emb in embs:
            result = find_by_voice(emb.tolist(), threshold=2.0)
            if result is not None:
                distances.append(float(result["_distance"]))
                labels.append(0)

    return distances, labels


def evaluate_n_speaker_enrolled(speaker_embeddings: dict, n_speaker_enrolled: int, threshold: float) -> dict | None:
    items = list(speaker_embeddings.items())
    enrolled  = dict(items[:n_speaker_enrolled])
    impostors = dict(items[n_speaker_enrolled:])

    clear_db()
    enroll(enrolled)
    distances, labels = collect_distances(enrolled, impostors)

    tp, tn, fp, fn = compute_confusion(labels, distances, threshold)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return {
        "n_speaker_enrolled": n_speaker_enrolled,
        "tpr_%": round(tpr * 100, 2),
        "fpr_%": round(fpr * 100, 2),
    }


def main():
    output_dir = Path("/tmp/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    speakers = load_speakers(max_files=10) # ist auf 10 Files begrenzt, damit der Benchmark nicht eine Stunde dauert sondern nur etwa 10 Minuten.
    embedder = VoiceEmbedder()
    embedder.load()
    speaker_embeddings = {}
    for speaker_id, files in tqdm(speakers.items(), desc="Embeddings"):
        speaker_embeddings[speaker_id] = [embedder.embed_from_pcm(*sf.read(f, dtype="float32")) for f in files]

    items     = list(speaker_embeddings.items())
    split_idx = len(items) // 2
    enrolled  = dict(items[:split_idx])   # z.B. {"id01": [emb0, emb1, ...], "id02": [...], ...}
    impostors = dict(items[split_idx:])   # z.B. {"id03": [emb0, emb1, ...], "id04": [...], ...}

    clear_db()
    enroll(enrolled)
    distances, labels = collect_distances(enrolled, impostors)
    threshold_result = find_best_threshold(distances, labels)
    best_threshold = threshold_result["threshold"]

    test_n_speaker = [1,2,3,4,5,7,10,13,16,20,25,30,35] # Anzahl der enrollten Sprecher, die getestet werden sollen. Insgesamt 40 Sprecher (VoxCeleb1 Test). Nicht enrollt = Impostor
    results_by_n = [] # Wird eine Liste von Dict
    for n_speaker_enrolled in test_n_speaker:
        r = evaluate_n_speaker_enrolled(speaker_embeddings, n_speaker_enrolled, best_threshold)
        if r is None:
            continue
        results_by_n.append(r)

    print(f"Bester Threshold : {best_threshold}, TPR={threshold_result['tpr_%']}%  FPR={threshold_result['fpr_%']}%")
    print(tabulate(results_by_n, headers="keys"))

    clear_db()


if __name__ == "__main__":
    main()
