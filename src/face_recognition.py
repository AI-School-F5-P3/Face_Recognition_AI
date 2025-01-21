from deepface import DeepFace
import cv2
import numpy as np
from typing import Tuple
from .config import FACE_RECOGNITION_SETTINGS, CAMERA_SETTINGS
from .utils import fetch_employee_by_id, log_access_attempt, get_timestamp, store_image, get_image
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId

class FacialRecognitionSystem:
    def __init__(self, db_client: MongoClient):
        # Initialize MongoDB client, GridFS, and settings
        self.db = db_client["face_recognition_db"]
        self.fs = GridFS(self.db)
        self.settings = FACE_RECOGNITION_SETTINGS
        self.model = DeepFace.build_model(self.settings["model_name"])

    def verify_face(self, face_frame: np.ndarray, reference_image_id: ObjectId) -> Tuple[bool, float]:
        try:
            print(f"Reference image ID: {reference_image_id}")
            reference_image = self.fs.get(reference_image_id).read()
            print(f"Reference image size: {len(reference_image)} bytes")

            result = DeepFace.verify(
                img1=face_frame,  # Pass the cropped face frame directly
                img2_bytes=reference_image,
                model_name=self.settings["model_name"],
                detector_backend=self.settings["detector_backend"],
                distance_metric=self.settings["distance_metric"],
                enforce_detection=False
            )
            print(f"Verification result: {result}")
            return result["verified"], result["distance"]
        except Exception as e:
            print(f"Error verifying face: {str(e)}")
            return False, 1.0


    def process_face_recognition(self, face_id: str, confidence: float) -> str:
        employee = fetch_employee_by_id(self.db, face_id)
        reference_image_id = employee.get("reference_image") if employee else None
        print(f"Employee fetched: {employee}")  # Debugging

        if employee:
            status = "ACCESS GRANTED"
            log_data = {
                "timestamp": get_timestamp(),
                "employee_id": employee["employee_id"],
                "name": employee["name"],
                "status": status,
                "confidence": confidence
            }
            log_access_attempt(self.db, log_data)
            return f"{employee['name']} ({status}) - {confidence:.2f}%"
        else:
            status = "ACCESS DENIED"
            log_data = {
                "timestamp": get_timestamp(),
                "employee_id": "Unknown",
                "name": "Unknown",
                "status": status,
                "confidence": confidence
            }
            log_access_attempt(self.db, log_data)
            return f"Unknown Person ({status}) - {confidence:.2f}%"

    def run(self):
        cap = cv2.VideoCapture(CAMERA_SETTINGS["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS["frame_height"])

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect faces
            faces = self.detect_faces(frame)

            for face_id, face_info in faces.items():
                x, y, w, h = face_info["coordinates"]
                face_frame = face_info["face"]

                # Lookup the employee's reference image ID
                employee = fetch_employee_by_id(self.db, face_id)
                reference_image_id = employee["image_id"] if employee and "image_id" in employee else None

                if reference_image_id:
                    verified, distance = self.verify_face(face_frame, reference_image_id)
                    confidence = 1.0 - distance
                else:
                    verified = False
                    confidence = 0.0  # Default confidence for unknown faces

                # Process recognition based on verification result
                message = self.process_face_recognition(face_id if verified else "Unknown", confidence)

                # Draw bounding box and label
                color = (0, 255, 0) if "ACCESS GRANTED" in message else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


    def detect_faces(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_data = {}
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y:y+h, x:x+w]
            face_data[str(i)] = {
                "face": face_crop,
                "coordinates": (x, y, w, h)  # Include coordinates for bounding boxes
            }

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle with thickness 2
            # Display a label
            cv2.putText(frame, f"Face {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return face_data


