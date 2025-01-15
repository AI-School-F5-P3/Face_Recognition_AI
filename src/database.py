import os
import pandas as pd
import cv2
from datetime import datetime, timedelta
import json
from typing import Dict, Optional, List, Any
from .utils import ensure_directory, get_timestamp
from .config import DATABASE_DIR, LOGS_DIR


class EmployeeDatabase:
    def __init__(self):
        """Initialize the employee database."""
        self.database_path = DATABASE_DIR
        self.metadata_file = os.path.join(DATABASE_DIR, "metadata.json")
        ensure_directory(DATABASE_DIR)
        self.metadata: Dict[str, Dict[str, str]] = {}
        self.load_metadata()

    # Add the validation methods at the top of the class
    def validate_employee_id(self, employee_id: str) -> bool:
        """Validate employee ID format."""
        return bool(employee_id and str(employee_id).strip())

    def validate_image(self, image: Any) -> bool:
        """Validate image format."""
        return (
            image is not None
            and hasattr(image, "shape")
            and len(image.shape) == 3
            and image.shape[2] in [3, 4]
        )

    def validate_name(self, name: str) -> bool:
        """Validate employee name."""
        return bool(name and name.strip() and len(name) <= 100)

    def load_metadata(self):
        """Load employee metadata from JSON file."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
                self.save_metadata()
        except json.JSONDecodeError as e:
            print(f"Error loading metadata: {str(e)}")
            self.metadata = {}
        except Exception as e:
            print(f"Unexpected error loading metadata: {str(e)}")
            self.metadata = {}

    def save_metadata(self) -> bool:
        """Save employee metadata to JSON file."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
            return False

    # Add backup functionality after save_metadata
    def backup_metadata(self) -> bool:
        """Create a backup of the metadata file."""
        try:
            backup_dir = os.path.join(self.database_path, "backups")
            ensure_directory(backup_dir)
            backup_file = os.path.join(
                backup_dir, f"metadata_backup_{get_timestamp()}.json"
            )
            with open(self.metadata_file, "r", encoding="utf-8") as src:
                with open(backup_file, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
            return True
        except Exception as e:
            print(f"Backup error: {str(e)}")
            return False

    def add_employee(self, employee_id: str, name: str, image: Any) -> bool:
        """Add new employee to database."""
        try:
            # Use validation methods
            if not self.validate_employee_id(employee_id):
                print("Invalid employee ID")
                return False

            if not self.validate_name(name):
                print("Invalid name")
                return False

            if not self.validate_image(image):
                print("Invalid image format")
                return False

            if str(employee_id) in self.metadata:
                print(f"Employee ID {employee_id} already exists")
                return False

            employee_dir = os.path.join(self.database_path, str(employee_id))
            ensure_directory(employee_dir)

            image_path = os.path.join(employee_dir, f"{get_timestamp()}.jpg")
            if not cv2.imwrite(image_path, image):
                print("Failed to save employee image")
                return False

            self.metadata[str(employee_id)] = {
                "name": name,
                "registration_date": get_timestamp(),
                "reference_image": image_path,
                "last_updated": datetime.now().isoformat(),
            }

            success = self.save_metadata()
            if success:
                # Create backup after successful addition
                self.backup_metadata()
            return success

        except Exception as e:
            print(f"Error adding employee: {str(e)}")
            return False


class AttendanceLogger:
    def __init__(self):
        """Initialize the attendance logger."""
        self.logs_path = LOGS_DIR
        ensure_directory(LOGS_DIR)
        self.current_log: List[Dict[str, str]] = []
        self.last_log_time: Dict[str, datetime] = {}

    def log_attendance(self, employee_id: str, name: str) -> bool:
        """Log attendance event with duplicate prevention."""
        try:
            current_time = datetime.now()

            if employee_id in self.last_log_time:
                time_diff = (
                    current_time - self.last_log_time[employee_id]
                ).total_seconds()
                if time_diff < 300:  # 5 minutes
                    return False

            entry = {
                "employee_id": employee_id,
                "name": name,
                "timestamp": current_time.isoformat(),
                "date": current_time.date().isoformat(),
                "time": current_time.time().isoformat(),
            }

            self.current_log.append(entry)
            self.last_log_time[employee_id] = current_time
            return True

        except Exception as e:
            print(f"Error logging attendance: {str(e)}")
            return False

    # Add cleanup method to AttendanceLogger
    def cleanup_old_logs(self, days: int = 30) -> bool:
        """Remove logs older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            self.current_log = [
                log
                for log in self.current_log
                if datetime.fromisoformat(log["timestamp"]) > cutoff_date
            ]
            return True
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
            return False

    def export_log(self) -> Optional[str]:
        """Export attendance log to CSV."""
        try:
            if not self.current_log:
                return None

            # Clean up old logs before export
            self.cleanup_old_logs()

            df = pd.DataFrame(self.current_log)
            filename = f"attendance_log_{get_timestamp()}.csv"
            filepath = os.path.join(self.logs_path, filename)

            df.to_csv(filepath, index=False)
            self.current_log = []  # Clear current log after export
            return filepath

        except Exception as e:
            print(f"Error exporting log: {str(e)}")
            return None
