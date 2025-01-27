"""API interface for the face detection system. Here's an overview:

route.py:


Uses FastAPI for the API implementation
Provides three endpoints:

/api/v1/detect: Detects faces in uploaded images
/api/v1/extract-features: Detects faces and extracts features
/api/v1/health: Health check endpoint


Handles image upload and processing
Includes proper error handling
Integrates with the core face detection and feature extraction modules


schemas.py:


Uses Pydantic for request/response validation
Defines schemas for:

DetectedFace: Face detection results
FaceDetectionResponse: Response for detection endpoint
FaceFeatures: Feature extraction results
FaceFeatureResponse: Response for feature extraction endpoint


Includes field validation and descriptions
Provides example responses

Key features:

Type validation and documentation using Pydantic
Proper error handling for image processing
JSON serialization of numpy arrays
Input validation for image files
Async endpoint handlers for better performance

To use this API, you'll need to:

Install dependencies: fastapi, python-multipart, pydantic
Run using an ASGI server like uvicorn
Send POST requests with image files to the endpoints"""