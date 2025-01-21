import cv2
from pymongo import MongoClient
from src.face_recognition import FacialRecognitionSystem
from src.database import EmployeeDatabase


def register_new_employee(db: EmployeeDatabase):
    """Utility function to register a new employee."""
    try:
        emp_id = input("Enter employee ID: ")
        if not db.validate_employee_id(emp_id):
            print("Employee ID already exists. Try another.")
            return
        name = input("Enter employee name: ")

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Press SPACE to capture image or Q to quit")

        while True:
            ret, frame = cap.read()

            # Validate frame
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                break

            # Verify frame is valid
            if not hasattr(frame, "shape"):
                print("Error: Invalid frame format")
                break

            cv2.imshow("Registration - Press SPACE to capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                # Verify frame again before processing
                if frame is not None and hasattr(frame, "shape"):
                    print(f"Image shape: {frame.shape if frame is not None else 'None'}, type: {type(frame)}")
                    if db.add_employee(emp_id, name, frame):
                        print(f"Successfully registered employee {name}")
                    else:
                        print("Failed to register employee")
                else:
                    print("Error: Invalid frame for processing")
                break

    except Exception as e:
        print(f"Error during registration: {str(e)}")
    finally:
        # Ensure cleanup happens even if an error occurs
        if "cap" in locals():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = EmployeeDatabase(client)

    while True:
        print("\nFacial Recognition System")
        print("1. Register new employee")
        print("2. Start recognition")
        print("3. Exit")

        choice = input("Enter choice (1-3): ")

        if choice == "1":
            register_new_employee(db)
        elif choice == "2":
            # Initialize FacialRecognitionSystem with the MongoDB connection
            system = FacialRecognitionSystem(client)
            system.run()
        elif choice == "3":
            break
        else:
            print("Invalid choice")
