Base Model Classes

# base_model.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models in the facial recognition system."""
    
    def __init__(self, model_path=None):
        self.model = self._create_model()
        if model_path:
            self.load_weights(model_path)
    
    @abstractmethod
    def _create_model(self):
        """Create and return the model architecture."""
        pass
    
    @abstractmethod
    def predict(self, input_data):
        """Make predictions on input data."""
        pass
    
    def load_weights(self, model_path):
        """Load model weights from file."""
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except Exception as e:
            raise Exception(f"Failed to load model weights from {model_path}: {str(e)}")
    
    def save_weights(self, model_path):
        """Save model weights to file."""
        try:
            torch.save(self.model.state_dict(), model_path)
        except Exception as e:
            raise Exception(f"Failed to save model weights to {model_path}: {str(e)}")
    
    def preprocess(self, input_data):
        """Default preprocessing method."""
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).float()
        return input_data
    
    def postprocess(self, output):
        """Default postprocessing method."""
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        return output