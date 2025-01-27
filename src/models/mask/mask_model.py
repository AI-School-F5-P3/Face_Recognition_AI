# mask/mask_model.py
import torch.nn as nn
from ..base_model import BaseModel

class MaskModel(BaseModel):
    def _create_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: mask/no mask
        )
        return model
    
    def predict(self, input_data):
        input_tensor = self.preprocess(input_data)
        mask_prob = self.model(input_tensor)
        return self.postprocess(mask_prob)