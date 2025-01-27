from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List
import io

from .schemas import (
    FaceDetectionResponse,
    FaceFeatureResponse,
    DetectedFace,
    FaceFeatures
)
from ..core.face_detector import FaceDetector
from ..core.feature_extractor import FaceFeatureExtractor

router = APIRouter(prefix="/api/v1")
face_detector = FaceDetector()
feature_extractor = FaceFeatureExtractor()

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(image: UploadFile = File(...)):
    """
    Detect faces in the uploaded image.
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
        # Detect faces
        faces = face_detector.detect_faces(img)
        
        # Convert to response format
        detected_faces = [
            DetectedFace(
                bbox=list(bbox),
                confidence=0.99  # Placeholder - implement actual confidence scoring
            )
            for bbox in faces
        ]
        
        return FaceDetectionResponse(
            num_faces=len(faces),
            faces=detected_faces
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-features", response_model=FaceFeatureResponse)
async def extract_face_features(image: UploadFile = File(...)):
    """
    Detect faces and extract features from the uploaded image.
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
        # Detect faces
        faces = face_detector.detect_faces(img)
        
        # Extract features for each face
        face_features = []
        for bbox in faces:
            # Get face ROI
            face_roi = face_detector.get_face_roi(img, bbox)
            if face_roi is not None:
                # Extract features
                features = feature_extractor.extract_features(face_roi)
                
                # Convert numpy arrays to lists for JSON serialization
                processed_features = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in features.items()
                }
                
                face_features.append(
                    FaceFeatures(
                        bbox=list(bbox),
                        features=processed_features
                    )
                )
        
        return FaceFeatureResponse(
            num_faces=len(faces),
            faces=face_features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    API health check endpoint.
    """
    return JSONResponse({"status": "healthy"})