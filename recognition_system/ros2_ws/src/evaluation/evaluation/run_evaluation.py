import soundfile as sf
import os
from tqdm import tqdm
from tabulate import tabulate

from microphone.voice_embedder import VoiceEmbedder
from evaluation.load_dataset import load_speakers, WAV_DIR_DOCKER_VOX1, WAV_DIR_DOCKER_VOX2
from evaluation.metrics import compute_confusion, find_best_threshold
from db.guest_store import find_by_voice, create_guest, clear_db

os.environ["DB_PATH"] = "/tmp/evaluation_db"  # Gilt nur für diesen Prozess, wird danach automatisch gelöscht


def embed_speakers(embedder: VoiceEmbedder, speakers: dict) -> dict:
    """
    Berechnet für jede Sprecher-ID die Embeddings aller zugehörigen Audiodateien.
    speakers kommt von load_speakers, z.B. {"id10001": ["/data/.../1.wav", "/data/.../2.wav"]}.
    Rückgabe im gleichen Aufbau, nur mit Embeddings statt Dateipfaden z.B. {"id10001": [array([...]), array([...])]}.
    """
    speaker_embeddings = {}
    for speaker_id, files in tqdm(speakers.items(), desc="Embeddings"): # desc setzt die Beschriftung vor der tqdm-Fortschrittsanzeige
        speaker_embeddings[speaker_id] = [embedder.embed_from_pcm(*sf.read(f, dtype="float32")) for f in files]
    return speaker_embeddings


def enroll(enrolled_embeddings: dict) -> None:
    """Legt für jeden Sprecher einen Gast mit dessen erstem Embedding als Enrollment-Template in der DB an."""
    for speaker_id, embs in enrolled_embeddings.items():
        create_guest(
            name=str(speaker_id),
            voice_emb=embs[0].tolist(), # nur erstes Embedding pro Sprecher in DB, Rest sind für Wiedererkennungstest
            episode_json="{}",
            memory_json="{}",
            text_embedder=lambda text: [0.0] * 384,
        )


def collect_distances(enrolled_embeddings: dict, impostor_embeddings: dict) -> tuple[list, list]:
    """
    Fragt für alle Test-Embeddings (echte Sprecher + Fremde) die DB ab und sammelt Distanzen mit Labels.
    Rückgabe: (distances, labels) als zwei gleich lange, parallele Listen, z.B. distances=[0.31, 1.42, ...],
    labels=[1, 0, ...].
    """
    distances = []
    labels = []

    # Echte Sprecher (Label 1): embs[1:] sind weitere Aufnahmen desselben Sprechers.
    # embs[0] fehlt hier absichtlich, da es schon in enroll() in der DB gelandet ist.
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


def evaluate_n_speaker_enrolled(speaker_embeddings: dict, n_speaker_enrolled: int, threshold: float) -> dict:
    """Enrollt n Sprecher in einer leeren DB und berechnet TPR/FPR gegen die restlichen Impostor-Sprecher."""
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
    """Kalibriert den Erkennungs-Threshold auf VoxCeleb1 und testet TPR/FPR bei steigender Sprecherzahl in der Datenbank auf VoxCeleb2."""
    embedder = VoiceEmbedder()
    embedder.load()

    # Threshold-Kalibrierung auf VoxCeleb1. Alle 40 Sprecher werden genutzt. Aber nur 10 Files pro Sprecher, um die Laufzeit der Evaluation zu reduzieren.
    calib_embeddings = embed_speakers(embedder, load_speakers(WAV_DIR_DOCKER_VOX1, max_files=10))

    # Hälfte der Sprecher als "enrollt" behandeln (bekannte Gäste), die andere Hälfte als Impostor (unbekannte Personen).
    calib_items     = list(calib_embeddings.items())
    calib_split_idx = len(calib_items) // 2
    calib_enrolled  = dict(calib_items[:calib_split_idx])
    calib_impostors = dict(calib_items[calib_split_idx:])

    clear_db()
    enroll(calib_enrolled)
    calib_distances, calib_labels = collect_distances(calib_enrolled, calib_impostors)
    threshold_result = find_best_threshold(calib_distances, calib_labels)
    best_threshold = threshold_result["threshold"]

    # VoxCeleb2 Test-Sprecher (118 Sprecher), ausschließlich für den die Evaluation mit n Speakern in Datenbank, um nicht
    # die gleichen Sprecher wie bei der Threshold-Kalibrierung zu nutzen. Sonst wäre die Evaluation nicht aussagekräftig, da die DB schon auf diese Sprecher trainiert wurde.
    speakers = load_speakers(WAV_DIR_DOCKER_VOX2, max_files=10) # ist auf 10 Files begrenzt, damit die Evaluation nicht zu lange dauert. Für die Evaluation reichen 10 Files pro Sprecher.
    speaker_embeddings = embed_speakers(embedder, speakers)

    test_n_speaker = [1,5,10,20,35,50,75,100] # Anzahl der enrollten Sprecher. Insgesamt 118 Sprecher (VoxCeleb2 Test). Nicht enrollt = Impostor
    results_by_n = [] # Wird eine Liste von Dict
    for n_speaker_enrolled in test_n_speaker:
        r = evaluate_n_speaker_enrolled(speaker_embeddings, n_speaker_enrolled, best_threshold)
        results_by_n.append(r)

    print(f"Bester Threshold : {best_threshold}, TPR={threshold_result['tpr_%']}%  FPR={threshold_result['fpr_%']}%  (kalibriert auf 40 VoxCeleb1-Sprecher)")
    print(tabulate(results_by_n, headers="keys")) # headers="keys" nimmt die Dict-Keys (n_speaker_enrolled, tpr_%, fpr_%) als Spaltenüberschriften

    clear_db()


if __name__ == "__main__":
    main()
