import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "data", "employee_database")
LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")

# Recognition settings
FACE_RECOGNITION_SETTINGS = {
    "model_name": "VGG-Face",  # or your preferred model
    "detector_backend": "opencv",  # or your preferred detector
    "distance_metric": "cosine",
    "threshold": 0.4,  # adjust based on your needs
}

CAMERA_SETTINGS = {
    'camera_id': 0,  # or your camera ID
    'frame_width': 640,
    'frame_height': 480
}

# Create required directories
for directory in [DATABASE_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
