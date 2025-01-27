# src/database/mongo/connection.py
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from typing import Optional
from ..schemas.models import Employee, AttendanceRecord
from ...utils.logging import get_logger

logger = get_logger(__name__)

class MongoDBConnection:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.connect()

    def connect(self):
        try:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            self.client = MongoClient(mongodb_url)
            self.db = self.client.facial_recognition
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def close(self):
        if self.client:
            self.client.close()