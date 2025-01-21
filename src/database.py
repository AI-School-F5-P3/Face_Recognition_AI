import os
from typing import Dict, Optional, Any
from bson.objectid import ObjectId
from pymongo import MongoClient
from gridfs import GridFS
from .utils import ensure_directory, get_timestamp, store_image
from .config import DATABASE_NAME
from datetime import datetime
import cv2
import numpy as np

class EmployeeDatabase:
    def __init__(self, db_client: MongoClient):
        """Initialize the employee database with MongoDB."""
        self.db = db_client[DATABASE_NAME]
        self.fs = GridFS(self.db)
        self.database_path = os.path.join(os.getcwd(), "data", "employee_database", "employee_images")
        ensure_directory(self.database_path)  # Ensure that the directory exists
        self.metadata: Dict[str, Dict[str, str]] = {}
        self.load_metadata()

    def load_metadata(self):
        """Load employee metadata from MongoDB."""
        try:
            # Fetch all employee metadata from MongoDB
            self.metadata = {
                employee["employee_id"]: employee
                for employee in self.db.employees.find()
            }
        except Exception as e:
            print(f"Error loading metadata from MongoDB: {str(e)}")
            self.metadata = {}

    def save_metadata(self) -> bool:
        """Save employee metadata to MongoDB."""
        try:
            # Save metadata for each employee
            for emp_id, employee_data in self.metadata.items():
                existing_employee = self.db.employees.find_one({"employee_id": emp_id})
                if existing_employee:
                    self.db.employees.update_one(
                        {"employee_id": emp_id},
                        {"$set": employee_data}
                    )
                else:
                    self.db.employees.insert_one(employee_data)
            return True
        except Exception as e:
            print(f"Error saving metadata to MongoDB: {str(e)}")
            return False

    def validate_employee_id(self, employee_id: str) -> bool:
        """Validate employee ID by checking if it already exists in the database."""
        existing_employee = self.db.employees.find_one({"employee_id": employee_id})
        if existing_employee:
            print(f"Employee ID {employee_id} already exists.")
            return False
        return True
    
    def validate_name(self, name: str) -> bool:
        """Validate employee name."""
        if not name or name.strip() == "":
            print("Employee name cannot be empty.")
            return False
        return True
    
    def validate_image(self, image: Any) -> bool:
        """Validate the image format."""
        if image is None:
            print("Image cannot be None.")
            return False

        # Check if image is a valid numpy array (valid frame from OpenCV)
        if not isinstance(image, (np.ndarray,)):
            print("Invalid image format.")
            return False

        return True

    def add_employee(self, employee_id: str, name: str, image: Any) -> bool:
        """Add new employee to database."""
        try:
            # Validate employee details
            if not self.validate_employee_id(employee_id):
                print("Invalid employee ID")
                return False

            if not self.validate_name(name):
                print("Invalid name")
                return False

            if not self.validate_image(image):
                print("Invalid image format")
                return False

            employee_dir = os.path.join(self.database_path, str(employee_id))
            ensure_directory(employee_dir)

            # Save the employee's reference image locally
            image_path = os.path.join(employee_dir, f"{get_timestamp()}.jpg")
            if image is None or image.size == 0:
                print("Error: Captured image is empty or invalid.")
                return False

            if not cv2.imwrite(image_path, image):
                print(f"Failed to save employee image at {image_path}")
                return False

            # Store the image in MongoDB using GridFS and retrieve its ObjectId
            image_id = store_image(image_path, f"{employee_id}_reference_image.jpg")

            # Save employee metadata
            self.metadata[str(employee_id)] = {
                "employee_id": employee_id,
                "name": name,
                "registration_date": get_timestamp(),
                "reference_image": image_id,  # Store image ObjectId (GridFS file ID)
                "last_updated": datetime.now().isoformat(),
            }

            # Save the metadata into MongoDB
            try:
                self.db.employees.insert_one({
                    "_id": str(employee_id),  # Set employee ID as the MongoDB document's _id
                    **self.metadata[str(employee_id)]  # Include all metadata fields
                })
            except Exception as e:
                print(f"Error saving employee to MongoDB: {str(e)}")
                return False

            print("Employee added successfully!")
            return True

        except Exception as e:
            print(f"Error adding employee: {str(e)}")
            return False

    def store_image(self, image: Any, filename: str) -> ObjectId:
        """Store an image in MongoDB's GridFS."""
        try:
            # Store the image in GridFS
            file_id = self.fs.put(image, filename=filename)
            return file_id
        except Exception as e:
            print(f"Error storing image in MongoDB: {str(e)}")
            return None

class AttendanceLogger:
    def __init__(self, db_client: MongoClient):
        """Initialize the attendance logger with MongoDB."""
        self.db = db_client[DATABASE_NAME]
        self.logs_collection = self.db.access_logs

    def log_attendance(self, employee_id: str, name: str) -> bool:
        """Log attendance event to MongoDB."""
        try:
            entry = {
                "employee_id": employee_id,
                "name": name,
                "timestamp": get_timestamp(),
                "date": get_timestamp("%Y-%m-%d"),
                "time": get_timestamp("%H:%M:%S"),
            }


            self.logs_collection.insert_one(entry)
            return True
        except Exception as e:
            print(f"Error logging attendance: {str(e)}")
            return False
