# crowd/crowd_model.py
import torch.nn as nn
from ..base_model import BaseModel

class CrowdModel(BaseModel):
    def _create_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)  # Single number for crowd count
        )
        return model
    
    def predict(self, input_data):
        input_tensor = self.preprocess(input_data)
        crowd_count = self.model(input_tensor)
        return self.postprocess(crowd_count)