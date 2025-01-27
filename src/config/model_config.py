from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class ModelConfig:
    enabled: bool
    model_path: str
    confidence_threshold: float
    additional_params: Dict[str, Any]

class ConfigurationManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        return {
            "face_recognition": {
                "enabled": True,
                "model_name": "VGG-Face",
                "detector_backend": "opencv",
                "distance_metric": "cosine",
                "threshold": 0.4,
            },
            "sentiment": {
                "enabled": True,
                "model_path": os.path.join(self.base_dir, "data", "trained_models", "sentiment"),
                "confidence_threshold": 0.5,
                "additional_params": {}
            },
            # Add configurations for other models