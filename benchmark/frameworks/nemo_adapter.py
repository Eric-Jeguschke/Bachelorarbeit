import torch
import numpy as np
import nemo.collections.asr as nemo_asr

class NeMo:
    def __init__(self, model: str, device: str) -> None:
        self.model = model
        self.device = device
        self.model_instance = None

    # https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/results.html#speaker-embedding-extraction
    def load(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model_instance = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.model, map_location=torch.device(self.device)) # Benutzt PyTorch intern
        self.model_instance.eval() # https://codesignal.com/learn/courses/building-a-neural-network-in-pytorch/lessons/evaluating-a-model-with-pytorch
        print(f"Modell NeMo {self.model} läuft auf: {self.device}")

    def embedding(self, path: str) -> np.ndarray:
        # get_embedding() gibt Tensor mit Form (1, emb_dim) zurück, .squeeze() → (emb_dim), .cpu().numpy() → NumPy-Array auf CPU
        return self.model_instance.get_embedding(path).squeeze().cpu().numpy()