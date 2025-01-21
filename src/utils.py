from pymongo import MongoClient
from gridfs import GridFS
from datetime import datetime
import os
import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any
from bson.objectid import ObjectId

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# MongoDB setup
def get_db():
    client = MongoClient("mongodb://localhost:27017")
    return client["face_recognition_db"]

db = get_db()
fs = GridFS(db)
    
def fetch_employee_by_id(db, employee_id: str) -> Optional[Dict]:
    """
    Fetch employee data from MongoDB based on employee_id.
    """
    try:
        return db.employees.find_one({"employee_id": employee_id})
    except Exception as e:
        logger.error(f"Error fetching employee by ID {employee_id}: {str(e)}")
        return None


def log_access_attempt(db, log_data: Dict) -> bool:
    """
    Log access attempt to MongoDB.
    """
    try:
        db.access_logs.insert_one(log_data)
        logger.info(f"Access attempt logged: {log_data}")
        return True
    except Exception as e:
        logger.error(f"Error logging access attempt: {str(e)}")
        return False

def get_image(file_id: ObjectId) -> Optional[np.ndarray]:
    """Generate a timestamp with a safe format for file names."""
    try:
        image_bytes = fs.get(file_id).read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error retrieving image with ID {file_id}: {str(e)}")
        return None


def store_image(file_path: str, filename: str) -> Optional[ObjectId]:
    """
    Store an image in MongoDB's GridFS.
    """
    try:
        with open(file_path, "rb") as f:
            file_id = fs.put(f, filename=filename)
        logger.info(f"Image stored with ID: {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"Error storing image: {str(e)}")
        return None

def get_image(file_id: ObjectId) -> Optional[bytes]:
    """
    Retrieve image from MongoDB's GridFS.
    """
    try:
        return fs.get(file_id).read()
    except Exception as e:
        logger.error(f"Error retrieving image with ID {file_id}: {str(e)}")
        return None

def ensure_directory(path: str) -> None:
    """Ensure that the given directory exists. If not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)