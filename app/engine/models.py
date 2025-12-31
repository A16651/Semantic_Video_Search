from threading import Lock
import torch
from transformers import SiglipModel, SiglipProcessor
from app.core.config import settings

class SigLipEngine:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SigLipEngine, cls).__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print(f"Loading SigLIP model: {settings.MODEL_NAME} on {settings.DEVICE}...")
        self.device = settings.DEVICE
        self.model = SiglipModel.from_pretrained(settings.MODEL_NAME).to(self.device)
        self.processor = SiglipProcessor.from_pretrained(settings.MODEL_NAME)
        self.model.eval()
        print("Model loaded successfully.")

    def get_components(self):
        return self.model, self.processor, self.device
