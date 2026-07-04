import torch
import soundfile as sf
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

class SpeechBrain:
    """Wrapper für die SpeechBrain-Sprecher-Embedding-Modelle"""

    def __init__(self, model: str, device: str) -> None:
        self.model_name = model
        self.device = device
        self.model_instance = None

    # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.speaker.html#speechbrain.inference.speaker.EncoderClassifier
    # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.utils.run_opts.html#speechbrain.utils.run_opts.RunOptions
    def load(self) -> None:
        """Lädt ein vortrainierte SpeechBrain-Model, welches bei __init__ fesgelegt wurde"""
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        device_name = "cuda:0" if self.device == "cuda" else self.device
        self.model_instance = EncoderClassifier.from_hparams(
            source=self.model_name,
            run_opts={"device": device_name}
        )
        self.model_instance.eval()
        print(f"Modell Speechbrain {self.model_name} läuft auf: {self.device}")

    # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    def embedding(self, path: str) -> np.ndarray:
        """Berechnet ein Sprecher-Embedding für eine Audiodatei"""
        data, _ = sf.read(path, dtype="float32") # torchAudio funktioniert nicht, deswegen ausgewichen auf soundfile
        waveform = torch.from_numpy(data).unsqueeze(0)
        embedding = self.model_instance.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()