from pathlib import Path

# VoxCeleb1 und VoxCeleb2 enthalten unterschiedliche Sprecher.
# run_evaluation.py nutzt eines der beiden Sets zur Threshold-Kalibrierung und das andere für den
# n_speaker_enrolled, damit nicht die Trainingsdaten als Testdaten genutzt werden.
WAV_DIR_DOCKER_VOX1 = Path("/data/voxceleb/vox1_test_wav")  # 40 Sprecher
WAV_DIR_DOCKER_VOX2 = Path("/data/voxceleb/vox2_test_wav")  # 118 Sprecher


def load_speakers(wav_dir: Path = WAV_DIR_DOCKER_VOX1, max_files: int = None) -> dict[str, list[str]]:
    """
    Durchsucht wav_dir nach Speaker-Unterordnern und gibt
    {speaker_id: [liste_von_wav_dateien]} zurück.
    Erwartet VoxCeleb-Struktur: wav_dir/speaker_id/**/*.wav
    """
    speakers = {}
    for speaker_dir in sorted(wav_dir.iterdir()): # sortieren für deterministische Reihenfolge
        files = sorted(speaker_dir.rglob("*.wav")) # rgblob für rekursives Finden, sortieren für deterministische Reihenfolge
        if max_files is not None:
            files = files[:max_files]
        file_paths = []
        for f in files:
            file_paths.append(str(f))
        speakers[speaker_dir.name] = file_paths
    return speakers
