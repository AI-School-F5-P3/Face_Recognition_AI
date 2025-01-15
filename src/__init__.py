def __init__(self):
    self.db = EmployeeDatabase()
    self.logger = AttendanceLogger()
    self.settings = FACE_RECOGNITION_SETTINGS
    # Remove or comment out the following line since we're not using the pre-loaded model
    # self.model = DeepFace.build_model(self.settings["model_name"])