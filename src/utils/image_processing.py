# src/utils/image_processing.py
import cv2
import numpy as np
from typing import Tuple, Optional
from .logging import get_logger

logger = get_logger(__name__)

class ImageProcessor:
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """Validate image format and dimensions."""
        try:
            return (
                image is not None
                and isinstance(image, np.ndarray)
                and len(image.shape) == 3
                and image.shape[2] in [1, 3, 4]
            )
        except Exception as e:
            logger.error(f"Error validating image format: {str(e)}")
            return False

    @staticmethod
    def save_image(image: np.ndarray, path: str, quality: Optional[int] = None) -> bool:
        """Save image with error handling and validation."""
        try:
            if not ImageProcessor.validate_image(image):
                logger.error("Invalid image data")
                return False

            params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(quality, 100))] if quality else []
            return cv2.imwrite(path, image, params)
        except Exception as e:
            logger.error(f"Error saving image to {path}: {str(e)}")
            return False

    @staticmethod
    def draw_face_box(
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        name: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding box and name on frame."""
        try:
            annotated_frame = frame.copy()
            x, y, w, h = face_location
            height, width = frame.shape[:2]

            # Validate coordinates
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                name, font, font_scale, font_thickness
            )

            text_x = max(0, min(x, width - text_width))
            text_y = max(text_height + baseline, y - 10)

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