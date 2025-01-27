# eye/eye_model.py
import torch.nn as nn
from ..base_model import BaseModel

class EyeModel(BaseModel):
    def _create_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Open/Closed classification
        )
        return model
    
    def predict(self, input_data):
        input_tensor = self.preprocess(input_data)
        eye_state = self.model(input_tensor)
        return self.postprocess(eye_state)