"""
Labels:
label: 1 = gleicher Sprecher, 0 = verschiedene Sprecher

Beispiel aus veri_test2.txt:
  1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6nnuk/00001.wav
  0 id10270/x6uYqmx31kE/00001.wav id10271/1gtz3xBMOZ0/00001.wav

https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf
"""

from pathlib import Path

# Pfad zum VoxCeleb-Datensatz (anpassen falls nötig)
VOXCELEB_ROOT = Path(r"C:\Studium\Semester 6\Bachelorarbeit\Datensatz\voxceleb")


def load_trials(trials_file_path: str, wav_folder_path: str) -> list[dict]:
    """
    Liest eine VoxCeleb Trial-Datei und gibt eine Liste von {label, audio1, audio2} zurück
    """
    trials = []
    wav = Path(wav_folder_path)

    with open(trials_file_path) as f:
        for row in f:
            split = row.strip().split()
            if len(split) < 3:
                continue # leere Zeilen oder unvollständige überspringen
            trials.append({
                "label": int(split[0]),
                "audio1": str(wav / split[1]), # absoluter Pfad zu Audio1
                "audio2": str(wav / split[2]), # absoluter Pfad zu Audio2
            })
    return trials


def load_voxceleb1_veri_test() -> list[dict]:
    """
    VoxCeleb1-O: Standard Benchmark
    """
    return load_trials(str(VOXCELEB_ROOT / "meta" / "veri_test2.txt"), str(VOXCELEB_ROOT / "vox1_test_wav"))


def load_voxceleb1_all_test() -> list[dict]:
    """
    VoxCeleb1-E: Extended Benchmark, ca. 15 Mal größer als Standard
    """
    return load_trials(str(VOXCELEB_ROOT / "meta" / "list_test_all2.txt"), str(VOXCELEB_ROOT / "vox1_test_wav"))
