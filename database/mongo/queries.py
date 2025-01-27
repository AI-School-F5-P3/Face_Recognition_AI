# src/database/mongo/queries.py
from typing import Dict, List, Optional, Any
from datetime import datetime
from .connection import MongoDBConnection
from ..schemas.models import Employee, AttendanceRecord
from ...utils.logging import get_logger

logger = get_logger(__name__)

class EmployeeDatabase:
    def __init__(self):
        self.connection = MongoDBConnection()
        self.db = self.connection.db
        self.employees = self.db.employees
        self.attendance = self.db.attendance

    async def add_employee(self, employee_data: Dict[str, Any]) -> bool:
        """Add new employee to database."""
        try:
            employee = Employee(**employee_data)
            result = await self.employees.insert_one(employee.dict())
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Error adding employee: {str(e)}")
            return False

    async def get_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve employee data by ID."""
        try:
            employee = await self.employees.find_one({"employee_id": employee_id})
            return employee if employee else None
        except Exception as e:
            logger.error(f"Error retrieving employee: {str(e)}")
            return None

    async def update_employee(self, employee_id: str, update_data: Dict[str, Any]) -> bool:
        """Update employee information."""
        try:
            update_data["last_updated"] = datetime.now()
            result = await self.employees.update_one(
                {"employee_id": employee_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating employee: {str(e)}")
            return False

    async def delete_employee(self, employee_id: str) -> bool:
        """Soft delete employee by setting is_active to False."""
        try:
            result = await self.employees.update_one(
                {"employee_id": employee_id},
                {"$set": {"is_active": False, "last_updated": datetime.now()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error deleting employee: {str(e)}")
            return False

class AttendanceLogger:
    def __init__(self):
        self.connection = MongoDBConnection()
        self.db = self.connection.db
        self.attendance = self.db.attendance

    async def log_attendance(self, attendance_data: Dict[str, Any]) -> bool:
        """Log attendance event."""
        try:
            record = AttendanceRecord(**attendance_data)
            result = await self.attendance.insert_one(record.dict())
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging attendance: {str(e)}")
            return False

    async def get_attendance_records(
        self,
        employee_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve attendance records with optional filters."""
        try:
            query = {}
            if employee_id:
                query["employee_id"] = employee_id
            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = self.attendance.find(query).sort("timestamp", -1)
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Error retrieving attendance records: {str(e)}")
            return []

    async def export_attendance_logs(self, start_date: datetime, end_date: datetime) -> str:
        """Export attendance logs to CSV."""
        try:
            records = await self.get_attendance_records(start_date=start_date, end_date=end_date)
            if not records:
                return None

            import pandas as pd
            df = pd.DataFrame(records)
            
            filename = f"attendance_log_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            export_path = os.path.join("data", "logs", filename)
            df.to_csv(export_path, index=False)
            
            return export_path
        except Exception as e:
            logger.error(f"Error exporting attendance logs: {str(e)}")
            return None