# src/pipeline/result_aggregator.py
from typing import Dict, Any
import numpy as np

class ResultAggregator:
    def __init__(self):
        self.results = {}

    def add_result(self, model_name: str, result: Dict[str, Any]):
        self.results[model_name] = result

    def get_combined_result(self) -> Dict[str, Any]:
        combined = {
            "face_recognition": self.results.get("face_recognition", {}),
            "additional_info": {}
        }

        # Add results from other models
        for model_name, result in self.results.items():
            if model_name != "face_recognition":
                combined["additional_info"][model_name] = result

        return combined

    def clear_results(self):
        self.results.clear()