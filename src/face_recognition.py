from deepface import DeepFace
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from .config import FACE_RECOGNITION_SETTINGS, CAMERA_SETTINGS
from .database import EmployeeDatabase, AttendanceLogger
from .utils import draw_face_box


class FacialRecognitionSystem:
    def __init__(self):
        self.db = EmployeeDatabase()
        self.logger = AttendanceLogger()
        self.settings = FACE_RECOGNITION_SETTINGS
        # Initialize DeepFace model once to improve performance
        self.model = DeepFace.build_model(self.settings["model_name"])

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validate if frame is properly formatted and not empty."""
        return (
            isinstance(frame, np.ndarray)
            and frame.size > 0
            and len(frame.shape) == 3
            and frame.shape[2] == 3
        )

    def verify_face(self, face_frame: np.ndarray, employee_data: Dict) -> bool:
        """
        Verify if face matches employee reference image.

        Args:
            face_frame: Numpy array containing the face image
            employee_data: Dictionary containing employee information

        Returns:
            bool: True if face matches, False otherwise
        """
        try:
            if not self._validate_frame(face_frame):
                return False

            # Convert face_frame to BGR if it's not already
            if face_frame.shape[2] == 4:  # RGBA format
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGBA2BGR)

            result = DeepFace.verify(
                img1_path=face_frame,
                img2_path=employee_data["reference_image"],
                model_name=self.settings[
                    "model_name"
                ],  # Changed from model to model_name
                detector_backend=self.settings["detector_backend"],
                distance_metric=self.settings["distance_metric"],
                enforce_detection=False,
            )
            return (
                result["verified"] and result["distance"] < self.settings["threshold"]
            )

        except Exception as e:
            print(f"Verification error: {str(e)}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process a single frame for face recognition.

        Args:
            frame: Input frame as numpy array

        Returns:
            Tuple containing processed frame and number of faces detected
        """
        if not self._validate_frame(frame):
            print("Invalid frame format")
            return frame, 0

        try:
            # Convert frame to RGB if necessary
            if frame.shape[2] == 4:  # RGBA format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Detect faces with error handling
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.settings["detector_backend"],
                enforce_detection=False,
                align=True,  # Enable face alignment for better accuracy
            )

            if not faces:
                return frame, 0

            faces_detected = 0
            for face in faces:
                region = face["facial_area"]
                # Validate region coordinates
                if not all(v >= 0 for v in region.values()):
                    continue

                # Ensure region doesn't exceed frame boundaries
                y1 = max(0, region["y"])
                y2 = min(frame.shape[0], region["y"] + region["h"])
                x1 = max(0, region["x"])
                x2 = min(frame.shape[1], region["x"] + region["w"])

                face_frame = frame[y1:y2, x1:x2]

                if face_frame.size == 0:
                    continue

                # Check against each known employee
                face_matched = False
                for emp_id, emp_data in self.db.metadata.items():
                    if self.verify_face(face_frame, emp_data):
                        self.logger.log_attendance(emp_id, emp_data["name"])
                        frame = draw_face_box(
                            frame, (x1, y1, x2 - x1, y2 - y1), emp_data["name"]
                        )
                        face_matched = True
                        faces_detected += 1
                        break

                if not face_matched:
                    frame = draw_face_box(
                        frame, (x1, y1, x2 - x1, y2 - y1), "Unknown", color=(0, 0, 255)
                    )
                    faces_detected += 1

            return frame, faces_detected

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame, 0

    def run(self):
        """Run real-time facial recognition."""
        cap = cv2.VideoCapture(CAMERA_SETTINGS["camera_id"])

        # Configure camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS["frame_height"])

        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")

        try:
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Failed to capture frame")
                    continue

                processed_frame, faces_detected = self.process_frame(frame)

                # Display frame with face count
                cv2.putText(
                    processed_frame,
                    f"Faces detected: {faces_detected}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Facial Recognition System", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("Stopping facial recognition system...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.export_log()
