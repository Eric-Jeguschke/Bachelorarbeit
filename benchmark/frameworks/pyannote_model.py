import os
import soundfile as sf
import numpy as np
import torch
from pyannote.audio import Model, Inference

class Pyannote:
    def __init__(self, model: str, device: str) -> None:
        self.model_name = model
        self.device = device
        self.token = os.environ.get("HUGGINGFACE_TOKEN")
        self.model_instance = None

    # https://huggingface.co/pyannote/embedding
    def load(self) -> None:
        model = Model.from_pretrained(self.model_name, token=self.token)
        model.eval()
        self.model_instance = Inference(model, window="whole", device=torch.device(self.device))
        print(f"Modell Pyannote {self.model_name} läuft auf: {self.device}")

    def embedding(self, path: str) -> np.ndarray:
        data, sample_rate = sf.read(path, dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, time)
        embedding = self.model_instance({"waveform": waveform, "sample_rate": sample_rate})
        return np.array(embedding).squeeze()
