import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
from src.face_recognition import FacialRecognitionSystem
from src.database import EmployeeDatabase
from src.utils import save_image, draw_face_box
import time

# Constants
METADATA_PATH = "data/employee_database/metadata.json"
PHOTOS_PATH = "data/employee_database/photos"

# Initialize database and recognition system
db = EmployeeDatabase()
recognition_system = FacialRecognitionSystem()

# Initialize session state variables
if "camera_open" not in st.session_state:
    st.session_state["camera_open"] = False
if "captured_frame" not in st.session_state:
    st.session_state["captured_frame"] = None
if "camera" not in st.session_state:
    st.session_state["camera"] = None

def initialize_camera():
    """Initialize the camera with optimal settings."""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            return False
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        st.session_state["camera"] = cap
        return True
    except Exception as e:
        st.error(f"Camera initialization error: {str(e)}")
        return False

def release_camera():
    """Safely release the camera."""
    try:
        if st.session_state["camera"] is not None:
            st.session_state["camera"].release()
            st.session_state["camera"] = None
        st.session_state["camera_open"] = False
    except Exception as e:
        st.error(f"Error closing camera: {str(e)}")

def read_frame():
    """Read a frame from the camera."""
    try:
        if st.session_state["camera"] is not None:
            ret, frame = st.session_state["camera"].read()
            if ret:
                return frame
    except Exception as e:
        st.error(f"Error reading frame: {str(e)}")
    return None

def save_employee_data(emp_id, name, image_path):
    """Save employee data to metadata.json"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
        
        # Load existing data
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Add new employee
        employee_data = {
            "id": str(emp_id),
            "name": str(name),
            "image_path": str(image_path),
            "reference_image": str(image_path)  # Add reference_image field
        }
        
        # Update or add employee
        for i, emp in enumerate(data):
            if emp.get("id") == str(emp_id):
                data[i] = employee_data
                break
        else:
            data.append(employee_data)

        # Save updated data
        with open(METADATA_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving employee data: {str(e)}")
        return False

def main():
    st.title("Facial Recognition System")

    menu = st.sidebar.selectbox("Menu", ["Home", "Register New Employee", "Start Recognition"])

    if menu == "Home":
        st.write("Welcome to the Facial Recognition System!")
        st.image("src/img/welcome.jpg", use_container_width=True)
        if st.session_state["camera_open"]:
            release_camera()

    elif menu == "Register New Employee":
        st.subheader("Register New Employee")

        emp_id = st.text_input("Enter Employee ID")
        name = st.text_input("Enter Employee Name")
        camera_frame = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Toggle Camera"):
                if not st.session_state["camera_open"]:
                    if initialize_camera():
                        st.session_state["camera_open"] = True
                else:
                    release_camera()

        if st.session_state["camera_open"]:
            frame = read_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
                with col2:
                    if st.button("Capture Photo"):
                        st.session_state["captured_frame"] = frame.copy()
                        st.success("Photo captured successfully!")

        if st.session_state["captured_frame"] is not None:
            display_frame = cv2.cvtColor(st.session_state["captured_frame"], cv2.COLOR_BGR2RGB)
            st.image(display_frame, caption="Captured Photo", width=300)
            
            if st.button("Register Employee"):
                if emp_id and name:
                    try:
                        os.makedirs(PHOTOS_PATH, exist_ok=True)
                        image_path = os.path.join(PHOTOS_PATH, f"{emp_id}.jpg")
                        
                        cv2.imwrite(image_path, st.session_state["captured_frame"])

                        if save_employee_data(emp_id, name, image_path):
                            if db.add_employee(emp_id, name, st.session_state["captured_frame"]):
                                st.success("Registration Complete!")
                                st.session_state["captured_frame"] = None
                            else:
                                st.error("Failed to update database.")
                        else:
                            st.error("Failed to save employee data.")
                    except Exception as e:
                        st.error(f"Error during registration: {str(e)}")
                else:
                    st.warning("Please enter both Employee ID and Name.")

    elif menu == "Start Recognition":
        st.subheader("Start Recognition")
        camera_frame = st.empty()
        status_placeholder = st.empty()

        if st.button("Toggle Recognition Camera"):
            if not st.session_state["camera_open"]:
                if initialize_camera():
                    st.session_state["camera_open"] = True
            else:
                release_camera()

        if st.session_state["camera_open"]:
            frame = read_frame()
            if frame is not None:
                try:
                    processed_frame, faces_detected = recognition_system.process_frame(frame)
                    
                    if processed_frame is not None:
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        camera_frame.image(display_frame, channels="RGB", use_container_width=True)

                        if faces_detected > 0:
                            for face in recognition_system.db.metadata.values():
                                if recognition_system.verify_face(frame, face):
                                    status_placeholder.success(f"ACCESO PERMITIDO: Bienvenido {face['name']}")
                                    st.image("src/img/door_open.gif")
                                    break
                            else:
                                status_placeholder.error("ACCESSO DENEGADO")
                        
                except Exception as e:
                    st.error(f"Recognition error: {str(e)}")
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_frame.image(display_frame, channels="RGB", use_container_width=True)

if __name__ == "__main__":
    main()