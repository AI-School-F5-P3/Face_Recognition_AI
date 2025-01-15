import cv2
import numpy as np
from datetime import datetime
import os
from typing import Tuple, Union, Optional
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> bool:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory to create

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        return False


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp string.

    Args:
        format: DateTime format string

    Returns:
        str: Formatted timestamp
    """
    try:
        return datetime.now().strftime(format)
    except Exception as e:
        logger.error(f"Error generating timestamp: {str(e)}")
        return datetime.now().strftime("%Y%m%d_%H%M%S")  # fallback format


def save_image(image: np.ndarray, path: str, quality: Optional[int] = None) -> bool:
    """
    Save image with error handling and validation.

    Args:
        image: NumPy array containing image data
        path: Path where image should be saved
        quality: JPEG quality (0-100), None for default

    Returns:
        bool: True if save was successful
    """
    try:
        # Validate image
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image data")
            return False

        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not ensure_directory(directory):
            return False

        # Save image with quality parameter if specified
        if quality is not None:
            params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(quality, 100))]
        else:
            params = []

        return cv2.imwrite(path, image, params)

    except Exception as e:
        logger.error(f"Error saving image to {path}: {str(e)}")
        return False


def draw_face_box(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    name: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding box and name on frame.

    Args:
        frame: Input image frame
        face_location: Tuple of (x, y, width, height)
        name: Name to display
        color: BGR color tuple
        thickness: Line thickness
        font_scale: Text size scale
        font_thickness: Text thickness

    Returns:
        np.ndarray: Frame with annotations
    """
    try:
        # Make a copy to avoid modifying original
        annotated_frame = frame.copy()

        # Validate inputs
        if not isinstance(face_location, tuple) or len(face_location) != 4:
            logger.error("Invalid face location format")
            return frame

        x, y, w, h = face_location

        # Validate coordinates
        height, width = frame.shape[:2]
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))

        # Draw rectangle
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)

        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            name, font, font_scale, font_thickness
        )

        # Ensure text stays within frame
        text_x = max(0, min(x, width - text_width))
        text_y = max(text_height + baseline, y - 10)

        # Draw text with background
        cv2.putText(
            annotated_frame,
            name,
            (text_x, text_y),
            font,
            font_scale,
            color,
            font_thickness,
        )

        return annotated_frame

    except Exception as e:
        logger.error(f"Error drawing face box: {str(e)}")
        return frame


def safe_delete_file(path: str) -> bool:
    """
    Safely delete a file with error handling.

    Args:
        path: Path to file to delete

    Returns:
        bool: True if deletion was successful or file didn't exist
    """
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        logger.error(f"Error deleting file {path}: {str(e)}")
        return False


def create_backup(source_path: str, backup_dir: str) -> Optional[str]:
    """
    Create a backup copy of a file.

    Args:
        source_path: Path to source file
        backup_dir: Directory to store backup

    Returns:
        Optional[str]: Path to backup file if successful, None otherwise
    """
    try:
        if not os.path.exists(source_path):
            logger.error(f"Source file {source_path} does not exist")
            return None

        if not ensure_directory(backup_dir):
            return None

        timestamp = get_timestamp()
        filename = os.path.basename(source_path)
        backup_path = os.path.join(
            backup_dir,
            f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}",
        )

        shutil.copy2(source_path, backup_path)
        return backup_path

    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        return None


def validate_image_format(image: np.ndarray) -> bool:
    """
    Validate image format and dimensions.

    Args:
        image: NumPy array containing image data

    Returns:
        bool: True if image format is valid
    """
    try:
        return (
            image is not None
            and isinstance(image, np.ndarray)
            and len(image.shape) == 3
            and image.shape[2] in [1, 3, 4]  # grayscale, BGR, or BGRA
        )
    except Exception as e:
        logger.error(f"Error validating image format: {str(e)}")
        return False
