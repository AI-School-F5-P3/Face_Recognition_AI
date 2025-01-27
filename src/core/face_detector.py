import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """Base class for face detection functionality."""
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection (0-1)
        """
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.min_confidence = min_confidence
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, width, height)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces.tolist()
    
    def get_face_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face region of interest (ROI) from image.
        
        Args:
            image: Input image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Face ROI image if valid bbox, None otherwise
        """
        if len(bbox) != 4:
            return None
            
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
            
        return image[y:y+h, x:x+w]
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw detected face bounding boxes on image.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            
        Returns:
            Image with drawn face boxes
        """
        image_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image_copy
    

    """ face_detector.py:
Contains the FaceDetector class for basic face detection
Uses OpenCV's Haar Cascade Classifier for face detection
Provides methods to detect faces, extract face ROIs, and draw bounding boxes
Includes type hints and documentation """