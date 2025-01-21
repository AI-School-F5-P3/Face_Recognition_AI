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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

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
if "last_frame_time" not in st.session_state:
    st.session_state["last_frame_time"] = 0
if "fps_array" not in st.session_state:
    st.session_state["fps_array"] = []


def initialize_camera():
    """Initialize the camera with optimal settings."""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            return False

        # Optimize camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)  # Set target FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Clear initial buffer
        for _ in range(5):
            cap.read()

        st.session_state["camera"] = cap
        st.session_state["last_frame_time"] = time.time()
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
        st.session_state["fps_array"] = []
    except Exception as e:
        st.error(f"Error closing camera: {str(e)}")


def read_frame():
    """Read and optimize frame from the camera."""
    try:
        if st.session_state["camera"] is not None:
            # Clear buffer
            st.session_state["camera"].grab()

            ret, frame = st.session_state["camera"].read()
            if ret:
                # Resize frame if needed
                if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # Compress frame
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, buffer = cv2.imencode(".jpg", frame, encode_param)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

                return frame
    except Exception as e:
        st.error(f"Error reading frame: {str(e)}")
    return None


def calculate_fps():
    """Calculate and update FPS."""
    current_time = time.time()
    fps = 1 / (current_time - st.session_state["last_frame_time"])
    st.session_state["last_frame_time"] = current_time

    # Keep a rolling average of FPS
    st.session_state["fps_array"].append(fps)
    if len(st.session_state["fps_array"]) > 30:
        st.session_state["fps_array"].pop(0)

    return np.mean(st.session_state["fps_array"])


def main():
    st.title("Facial Recognition System")

    # Debug mode checkbox in sidebar
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")
    if debug_mode:
        debug_container = st.sidebar.container()
        fps_placeholder = debug_container.empty()
        frame_info_placeholder = debug_container.empty()

    menu = st.sidebar.selectbox(
        "Menu", ["Home", "Register New Employee", "Start Recognition"]
    )

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
            if st.button("Toggle Camera", key="register_toggle_camera"):
                if not st.session_state["camera_open"]:
                    if initialize_camera():
                        st.session_state["camera_open"] = True
                else:
                    release_camera()

        if st.session_state["camera_open"]:
            while True:
                frame = read_frame()
                if frame is not None:
                    # Calculate FPS
                    if debug_mode:
                        fps = calculate_fps()
                        fps_placeholder.text(f"FPS: {fps:.2f}")
                        frame_info_placeholder.text(
                            f"Frame Size: {frame.shape}\n"
                            f"Frame Type: {frame.dtype}\n"
                            f"Memory Usage: {frame.nbytes / 1024:.2f} KB"
                        )

                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_frame.image(
                        frame_rgb, channels="RGB", use_container_width=True
                    )

                    # Control frame rate
                    time_elapsed = time.time() - st.session_state["last_frame_time"]
                    if time_elapsed < 1.0 / TARGET_FPS:
                        time.sleep(1.0 / TARGET_FPS - time_elapsed)

                    with col2:
                        if st.button("Capture Photo", key="capture_photo"):
                            st.session_state["captured_frame"] = frame.copy()
                            st.success("Photo captured successfully!")
                            break
                else:
                    st.error("Failed to read frame from camera")
                    break

        if st.session_state["captured_frame"] is not None:
            display_frame = cv2.cvtColor(
                st.session_state["captured_frame"], cv2.COLOR_BGR2RGB
            )
            st.image(display_frame, caption="Captured Photo", width=300)

            if st.button("Register Employee", key="register_employee"):
                if emp_id and name:
                    try:
                        os.makedirs(PHOTOS_PATH, exist_ok=True)
                        image_path = os.path.join(PHOTOS_PATH, f"{emp_id}.jpg")

                        cv2.imwrite(image_path, st.session_state["captured_frame"])

                        if save_employee_data(emp_id, name, image_path):
                            if db.add_employee(
                                emp_id, name, st.session_state["captured_frame"]
                            ):
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

        if st.button("Toggle Recognition Camera", key="recognition_toggle_camera"):
            if not st.session_state["camera_open"]:
                if initialize_camera():
                    st.session_state["camera_open"] = True
            else:
                release_camera()

        if st.session_state["camera_open"]:
            while True:
                frame = read_frame()
                if frame is not None:
                    try:
                        # Calculate FPS for debug mode
                        if debug_mode:
                            fps = calculate_fps()
                            fps_placeholder.text(f"FPS: {fps:.2f}")
                            frame_info_placeholder.text(
                                f"Frame Size: {frame.shape}\n"
                                f"Frame Type: {frame.dtype}\n"
                                f"Memory Usage: {frame.nbytes / 1024:.2f} KB"
                            )

                        processed_frame, faces_detected = (
                            recognition_system.process_frame(frame)
                        )

                        if processed_frame is not None:
                            display_frame = cv2.cvtColor(
                                processed_frame, cv2.COLOR_BGR2RGB
                            )
                            camera_frame.image(
                                display_frame, channels="RGB", use_container_width=True
                            )

                            if faces_detected > 0:
                                for face in recognition_system.db.metadata.values():
                                    if recognition_system.verify_face(frame, face):
                                        status_placeholder.success(
                                            f"ACCESO PERMITIDO: Bienvenido {face['name']}"
                                        )
                                        st.image("src/img/door_open.gif")
                                        break
                                else:
                                    status_placeholder.error("ACCESSO DENEGADO")

                        # Control frame rate
                        time_elapsed = time.time() - st.session_state["last_frame_time"]
                        if time_elapsed < 1.0 / TARGET_FPS:
                            time.sleep(1.0 / TARGET_FPS - time_elapsed)

                    except Exception as e:
                        st.error(f"Recognition error: {str(e)}")
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_frame.image(
                            display_frame, channels="RGB", use_container_width=True
                        )
                else:
                    st.error("Failed to read frame from camera")
                    break


if __name__ == "__main__":
    main()
