# age/age_model.py
import torch.nn as nn
from ..base_model import BaseModel

class AgeModel(BaseModel):
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
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 101)  # Age prediction from 0-100
        )
        return model
    
    def predict(self, input_data):
        input_tensor = self.preprocess(input_data)
        age_distribution = self.model(input_tensor)
        predicted_age = self.postprocess(age_distribution).argmax()
        return predicted_age