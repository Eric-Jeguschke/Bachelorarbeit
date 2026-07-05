import numpy as np
import torch
from pyannote.audio import Model, Inference

# pyannote/wespeaker-voxceleb-resnet34-LM speaker embedding model
# Quelle: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM


class VoiceEmbedder:
    """Wrapper für das Pyannote-Wespeaker-Stimm-Embedding-Modell."""

    def __init__(self) -> None:
        self.model_name = "pyannote/wespeaker-voxceleb-resnet34-LM"
        self.device = "cpu" # da Code später im Optimalfall auf PIB Roboter laufen soll und der kein CUDA hat, erstmal mit CPU testen und Performance beobachten
        self.model_instance = None

    def load(self) -> None:
        """Lädt das vortrainierte Modell und führt einen Warmup-Durchlauf aus."""
        model = Model.from_pretrained(self.model_name)
        self.model_instance = Inference(model, window="whole", device=torch.device(self.device))
        # PyTorch Warmup beim ersten Aufruf triggern, damit echte Aufnahmen sofort schnell sind
        self.embed_from_pcm(np.zeros(16000, dtype=np.float32), 16000)

    def embed_from_pcm(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Berechnet das Stimm-Embedding aus PCM-Rohdaten.
        PCM (Pulse-Code Modulation) ist die unkomprimierte, digitale Darstellung eines Audiosignals: Die Audio
        wird in festen Zeitabständen abgetastet und als Zahlenfolge gespeichert.
        """
        # Stille wird bereits entfernt, bevor pcm hier ankommt
        pcm = pcm.squeeze()  # sounddevice gibt (samples, channels) zurück -> squeeze zu (samples,)
        waveform = torch.from_numpy(pcm).unsqueeze(0) # (samples,) -> (1, samples) = (channels, samples) wie pyannote es erwartet
        emb = self.model_instance({"waveform": waveform, "sample_rate": sample_rate})
        return np.array(emb).squeeze()
