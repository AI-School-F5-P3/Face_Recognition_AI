from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any

class DetectedFace(BaseModel):
    """Schema for a detected face."""
    bbox: List[int] = Field(
        ...,
        description="Bounding box coordinates [x, y, width, height]",
        min_items=4,
        max_items=4
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score",
        ge=0.0,
        le=1.0
    )

class FaceDetectionResponse(BaseModel):
    """Response schema for face detection endpoint."""
    num_faces: int = Field(..., description="Number of faces detected")
    faces: List[DetectedFace] = Field(..., description="List of detected faces")

class FaceFeatures(BaseModel):
    """Schema for face features."""
    bbox: List[int] = Field(
        ...,
        description="Bounding box coordinates [x, y, width, height]",
        min_items=4,
        max_items=4
    )
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of extracted features"
    )

class FaceFeatureResponse(BaseModel):
    """Response schema for face feature extraction endpoint."""
    num_faces: int = Field(..., description="Number of faces detected")
    faces: List[FaceFeatures] = Field(
        ...,
        description="List of faces with their extracted features"
    )

    class Config:
        schema_extra = {
            "example": {
                "num_faces": 1,
                "faces": [
                    {
                        "bbox": [100, 100, 200, 200],
                        "features": {
                            "landmarks": [[x, y] for x, y in zip(range(5), range(5))],
                            "embedding": [0.1] * 128,
                            "histogram": [0.0] * 948
                        }
                    }
                ]
            }
        }