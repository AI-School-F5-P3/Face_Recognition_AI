# src/utils/validation.py
from typing import Any, Dict
import os
from datetime import datetime
from .logging import get_logger

logger = get_logger(__name__)

class Validator:
    @staticmethod
    def validate_directory(directory: str) -> bool:
        """Validate and create directory if it doesn't exist."""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            return False

    @staticmethod
    def validate_employee_data(data: Dict[str, Any]) -> bool:
        """Validate employee data structure."""
        required_fields = ["employee_id", "name"]
        return all(field in data for field in required_fields)

    @staticmethod
    def validate_employee_id(employee_id: str) -> bool:
        """Validate employee ID format."""
        return bool(employee_id and str(employee_id).strip())

    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate employee name."""
        return bool(name and name.strip() and len(name) <= 100)

    @staticmethod
    def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
        """Get current timestamp string."""
        try:
            return datetime.now().strftime(format)
        except Exception as e:
            logger.error(f"Error generating timestamp: {str(e)}")
            return datetime.now().strftime("%Y%m%d_%H%M%S")