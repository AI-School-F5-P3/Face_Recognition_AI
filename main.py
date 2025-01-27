# main.py
import cv2
import numpy as np
from src.config.model_config import ConfigurationManager
from src.pipeline.model_pipeline import ModelPipeline
from src.pipeline.result_aggregator import ResultAggregator
from src.database import EmployeeDatabase
from src.utils.visualization import draw_results

class FacialAnalysisSystem:
    def __init__(self):
        self.config = ConfigurationManager()._load_config()
        self.pipeline = ModelPipeline(self.config)
        self.result_aggregator = ResultAggregator()
        self.db = EmployeeDatabase()

    def register_new_employee(self):
        """Utility function to register a new employee."""
        try:
            emp_id = input("Enter employee ID: ")
            name = input("Enter employee name: ")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera")

            print("Press SPACE to capture image or Q to quit")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                cv2.imshow("Registration - Press SPACE to capture", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    if self.db.add_employee(emp_id, name, frame):
                        print(f"Successfully registered employee {name}")
                    else:
                        print("Failed to register employee")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_analysis(self):
        """Run real-time facial analysis with all enabled models."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Process frame through all enabled models
                results = self.pipeline.process_frame(frame)
                self.result_aggregator.add_result("frame_results", results)
                
                # Get combined results and visualize
                combined_results = self.result_aggregator.get_combined_result()
                display_frame = draw_results(frame, combined_results)

                cv2.imshow("Facial Analysis System", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.result_aggregator.clear_results()

if __name__ == "__main__":
    system = FacialAnalysisSystem()
    
    while True:
        print("\nFacial Analysis System")
        print("1. Register new employee")
        print("2. Start analysis")
        print("3. Exit")

        choice = input("Enter choice (1-3): ")

        if choice == "1":
            system.register_new_employee()
        elif choice == "2":
            system.run_analysis()
        elif choice == "3":
            break
        else:
            print("Invalid choice")