# src/database/schemas/models.py
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class Employee(BaseModel):
    employee_id: str = Field(..., description="Unique employee identifier")
    name: str = Field(..., description="Employee full name")
    registration_date: datetime = Field(default_factory=datetime.now)
    reference_image_path: str = Field(..., description="Path to employee's reference image")
    last_updated: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)
    metadata: dict = Field(default_factory=dict)

class AttendanceRecord(BaseModel):
    employee_id: str
    name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    entry_type: str = Field(default="entry")  # entry/exit
    location: Optional[str] = None
    device_id: Optional[str] = None
