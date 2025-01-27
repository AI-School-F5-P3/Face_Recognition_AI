import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from .face_detector import FaceDetector


class FaceFeatureExtractor:
    """Extract facial features from detected faces."""

    def __init__(self):
        """Initialize the feature extractor."""
        # Initialize face detector
        self.face_detector = FaceDetector()

        # Load facial landmark detector
        self.landmark_detector = cv2.face.createFacemark()
        self.landmark_detector.loadModel("models/facial_landmarks.dat")

    def extract_features(self, face_roi: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from face region of interest.

        Args:
            face_roi: Face region image

        Returns:
            Dictionary containing extracted features:
            - 'landmarks': Facial landmark points
            - 'embedding': Face embedding vector
            - 'histogram': Face histogram features
        """
        features = {}

        # Get facial landmarks
        success, landmarks = self.landmark_detector.fit(face_roi)
        if success:
            features["landmarks"] = landmarks[0]

        # Extract face embedding (using a pre-trained model)
        embedding = self._compute_face_embedding(face_roi)
        if embedding is not None:
            features["embedding"] = embedding

        # Compute histogram features
        features["histogram"] = self._compute_histogram_features(face_roi)

        return features

    def _compute_face_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute face embedding vector using deep learning model.

        Args:
            face_roi: Face region image

        Returns:
            Face embedding vector if successful, None otherwise
        """
        try:
            # Preprocess image
            blob = cv2.dnn.blobFromImage(
                face_roi,
                scalefactor=1.0 / 255,
                size=(96, 96),
                mean=[0.485, 0.456, 0.406],
                swapRB=True,
            )

            # TODO: Load and use actual face embedding model
            # This is a placeholder that returns random embedding
            return np.random.rand(128)

        except Exception as e:
            print(f"Error computing face embedding: {str(e)}")
            return None

    def _compute_histogram_features(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Compute histogram features from face image.

        Args:
            face_roi: Face region image

        Returns:
            Concatenated histogram features
        """
        # Convert to different color spaces
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

        # Compute histograms
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        # Normalize histograms
        cv2.normalize(hist_gray, hist_gray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Concatenate all histogram features
        return np.concatenate([hist_gray, hist_h, hist_s, hist_v])

    """feature_extractor.py:

Contains the FaceFeatureExtractor class that builds on the face detector
Extracts multiple types of facial features:

Facial landmarks using OpenCV's Facemark
Face embeddings (placeholder for deep learning model)
Histogram features in multiple color spaces

Includes proper error handling and type hints"""


""" Some notes and TODOs:

You'll need to install OpenCV (cv2) and NumPy
The face embedding model is currently a placeholder - you'll need to add your preferred model
The facial landmark model path needs to be updated to your actual model file
You might want to add additional feature extraction methods based on your needs"""
