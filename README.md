# Face Recognition project - Employee Management System

### Business Presentation

https://www.canva.com/design/DAGc0I0yjQE/dTzLm4posF9WEMH86vnpAA/edit?utm_content=DAGc0I0yjQE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

### Overview

This project is an Employee Management System designed to store employee information, manage their attendance, and verify employee identities using facial recognition. The application integrates with MongoDB to store metadata and images and leverages OpenCV and DeepFace for image processing and facial recognition.

### Features

Add and manage employee records with unique IDs, names, and reference images.

Store employee images securely in MongoDB's GridFS.

Verify employee identity using facial recognition.

Log employee attendance with timestamps.

### Technologies Used

#### Backend

- Python: Main programming language for the application.

- PyMongo: For interacting with the MongoDB database.

- GridFS: To handle large file storage (employee images).

#### Libraries and Tools

- OpenCV: For image handling and processing.

- DeepFace: For facial recognition.

- NumPy: For numerical operations and image decoding.

#### Database

- MongoDB: Stores employee metadata, attendance logs, and images.

### Setup Instructions

Follow these steps to deploy the project on your local machine:

Prerequisites:

- Python (3.8 or later).

- MongoDB (installed locally or accessible remotely).

- Virtual Environment (recommended).

Steps: 

1. Clone the Repository

$ git clone https://github.com/AI-School-F5-P3/Face_Recognition_AI.git
$ cd employee-management-system

2. Create a Virtual Environment and Activate It

$ python -m venv venv
# For Windows:
$ .\venv\Scripts\activate
# For MacOS/Linux:
$ source venv/bin/activate

3. Install Dependencies

$ pip install -r requirements.txt

4. Configure MongoDB

Ensure MongoDB is running locally or accessible remotely. Update the config.py file to set the correct DATABASE_NAME and connection URI.

Example config.py:

DATABASE_NAME = "employee_db"
MONGO_URI = "mongodb://localhost:27017/"

5. Run the Application

$ python main.py

The application should now be running on http://localhost:5000.

### Client Brief

The client required an intuitive system to:

- Register Employees: Each employee should have a unique ID, name, and reference image stored securely.

- Verify Attendance: Use facial recognition to verify employees as they clock in and out.

- Secure Image Storage: Handle large image files without performance degradation.

The system was built to ensure high accuracy, scalability, and compatibility with future enhancements like multi-factor authentication or mobile app integration.

