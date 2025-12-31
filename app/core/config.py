import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Semantic Video Search"
    API_V1_STR: str = "/api/v1"
    
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "video_embeddings")

    # Model
    MODEL_NAME: str = "google/siglip-base-patch16-224"
    USE_CUDA: bool = True
    
    @property
    def DEVICE(self) -> str:
        if self.USE_CUDA:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return "cpu"

    class Config:
        env_file = ".env"

settings = Settings()
