from ..base_model import BaseModel
from deepface import DeepFace
import numpy as np
from typing import Dict, Any

class FaceRecognitionModel(BaseModel):
    def _load_model(self):
        self.model = DeepFace.build_model(self.config["model_name"])
        self.detector_backend = self.config["detector_backend"]
        self.distance_metric = self.config["distance_metric"]
        self.threshold = self.config["threshold"]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 4:  # RGBA format
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        image = self.preprocess(image)
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=True
        )
        return {"faces": faces}