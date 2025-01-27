# pose/pose_model.py
import torch.nn as nn
from ..base_model import BaseModel

class PoseModel(BaseModel):
    def _create_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 68 * 2)  # 68 landmarks, 2 coordinates each
        )
        return model
    
    def predict(self, input_data):
        input_tensor = self.preprocess(input_data)
        landmarks = self.model(input_tensor)
        return self.postprocess(landmarks.reshape(-1, 68, 2))