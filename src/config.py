import os
from pymongo import MongoClient

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "data", "employee_database")
LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")

# Recognition settings
FACE_RECOGNITION_SETTINGS = {
    "model_name": "VGG-Face",
    "detector_backend": "opencv",
    "distance_metric": "cosine",
    "threshold": 0.4,
}

CAMERA_SETTINGS = {
    'camera_id': 0,
    'frame_width': 640,
    'frame_height': 480
}

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"  # Update if using remote or Atlas
DATABASE_NAME = "employee_management"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Create required directories
for directory in [DATABASE_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

