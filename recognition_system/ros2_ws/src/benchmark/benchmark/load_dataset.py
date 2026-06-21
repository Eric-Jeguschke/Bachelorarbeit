"""
Labels:
1 = gleicher Sprecher, 0 = verschiedene Sprecher

Beispiel aus veri_test2.txt:
1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6nnuk/00001.wav
0 id10270/x6uYqmx31kE/00001.wav id10271/1gtz3xBMOZ0/00001.wav

Paper über den Datensatz: https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf
"""

from pathlib import Path

VOXCELEB_DIR = Path(r"C:\Studium\Semester 6\Bachelorarbeit\Datensatz\voxceleb")
WAV_DIR_DOCKER = Path("/data/voxceleb/vox1_test_wav")


def load_voxceleb1_veri_test() -> list[dict]:
    """
    VoxCeleb1-O: Standard Benchmark.
    Liest eine VoxCeleb Trial-Datei und gibt eine Liste von {label, audio1, audio2} zurück
    """
    trials = []
    wav_path = VOXCELEB_DIR / "vox1_test_wav"

    with open(VOXCELEB_DIR / "meta" / "veri_test2.txt") as f:
        for row in f:
            split = row.strip().split()
            if len(split) < 3:
                continue # leere Zeilen oder unvollständige überspringen
            trials.append({
                "label": int(split[0]),
                "audio1": str(wav_path / split[1]), # absoluter Pfad zu Audio1
                "audio2": str(wav_path / split[2]), # absoluter Pfad zu Audio2
            })
    return trials


def load_speakers(max_files: int = None) -> dict[str, list[str]]:
    """
    Durchsucht WAV_DIR nach Speaker-Unterordnern und gibt
    {speaker_id: [liste_von_wav_dateien]} zurück.
    Erwartet VoxCeleb-Struktur: WAV_DIR/speaker_id/**/*.wav
    """
    speakers = {}
    for speaker_dir in sorted(WAV_DIR_DOCKER.iterdir()): # sortieren für deterministische Reihenfolge
        files = sorted(speaker_dir.rglob("*.wav")) # rgblob für rekursives Finden, sortieren für deterministische Reihenfolge
        if max_files is not None:
            files = files[:max_files]
        file_paths = []
        for f in files:
            file_paths.append(str(f))
        speakers[speaker_dir.name] = file_paths
    return speakers
