# pyannote.audio Speaker Embedding Adapter
#
# pyannote.audio 3.x: Bredin & Laurent (2023)
#   "End-to-end speaker segmentation for overlap-aware resegmentation"
#   https://arxiv.org/abs/2104.04045
#
# Modell "pyannote/embedding":
#   - Braucht einen HuggingFace-Token (kostenlos): https://huggingface.co/pyannote/embedding
#   - Token übergeben via hf_token-Parameter ODER Umgebungsvariable HUGGINGFACE_TOKEN
#   - Zugang zum Modell einmalig auf HuggingFace beantragen (Klick auf "Agree and access repository")
#
# Installieren: pip install pyannote.audio

import os
import numpy as np
import torch
import soundfile as sf
from pyannote.audio import Model, Inference

class Pyannote:
    def __init__(self, model: str, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.token = os.environ.get("HUGGINGFACE_TOKEN")
        self.inference = None

    # https://huggingface.co/pyannote/embedding
    def load(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        model = Model.from_pretrained(self.model, token=self.token)
        self.inference = Inference(model, window="whole", device=torch.device(self.device))
        print(f"Modell Pyannote läuft auf: {self.device}")

    def embedding(self, path: str) -> np.ndarray:
        emb = self.inference(path)
        return np.array(emb).squeeze() # .squeeze(), damit aus (1, emb_dim) → (emb_dim) wird
