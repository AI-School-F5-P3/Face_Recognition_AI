from typing import Dict, List, Any
import numpy as np
from ..models.face_recognition.recognition_model import FaceRecognitionModel
from ..models.sentiment.sentiment_model import SentimentModel
# Import other models as needed

class ModelPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = self._initialize_models()

    def _initialize_models(self) -> Dict[str, BaseModel]:
        models = {}
        if self.config["face_recognition"]["enabled"]:
            models["face_recognition"] = FaceRecognitionModel(
                self.config["face_recognition"]
            )
        if self.config["sentiment"]["enabled"]:
            models["sentiment"] = SentimentModel(
                self.config["sentiment"]
            )
        # Initialize other models based on config
        return models

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        results = {}
        for model_name, model in self.models.items():
            try:
                results[model_name] = model.predict(frame)
            except Exception as e:
                print(f"Error in {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        return results
