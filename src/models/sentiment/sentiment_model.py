from ..base_model import BaseModel
from deepface import DeepFace
import numpy as np
from typing import Dict, Any

class SentimentModel(BaseModel):
    def _load_model(self):
        self.model = DeepFace.build_model('Emotion')

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Add any specific preprocessing for sentiment analysis
        return image

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            analysis = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False
            )
            return {"sentiment": analysis[0]["emotion"]}
        except Exception as e:
            return {"sentiment": None, "error": str(e)}