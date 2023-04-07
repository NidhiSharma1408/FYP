
import json
from datetime import datetime, timedelta
from core.models import AttendanceModel
from django.utils import timezone
class Attendance:
    FILENAME = 'attendance.json'  # File name for JSON storage

    def __init__(self):
        self.attendance_data = []

    def write_attendance(self, name):
        self.load_attendance_data()  # Load attendance data from JSON
        if not self.is_name_present(name):  # Check if name is already present
            attendance_record = {'name': name, 'timestamp': str(timezone.now())}
            self.attendance_data.append(attendance_record)  # Add attendance record
            self.save_attendance_data()  # Save attendance data to JSON

    def is_name_present(self, name):
        for record in self.attendance_data:
            if record['name'] == name:
                return True
        return False

    def load_attendance_data(self):
        try:
            with open(self.FILENAME, 'r') as file:
                self.attendance_data = json.load(file)
        except FileNotFoundError:
            pass

    def save_attendance_data(self):
        with open(self.FILENAME, 'w') as file:
            json.dump(self.attendance_data, file)

    def write_attendance_to_db(self):
        print("Writing attendance to db...")
        self.load_attendance_data()  # Load attendance data from JSON
        if self.attendance_data:
            for record in self.attendance_data:
                attendance = AttendanceModel(name=record['name'], timestamp=record['timestamp'])
                attendance.save()  # Save attendance record to DB
            self.attendance_data = []  # Clear attendance data in JSON after writing to DB
            self.save_attendance_data()  # Save updated attendance data to JSON
        print("Written to db")

    def schedule_write_attendance_to_db(self, root):
        self.write_attendance_to_db()  # Write attendance data to DB
        # Schedule next write to DB after 30 minutes
        now = datetime.now()
        next_write_time = now + timedelta(minutes=1)
        self.schedule_after(next_write_time, root)

    def schedule_after(self, dt, root):
        current_time = datetime.now()
        delta = dt - current_time
        if delta.total_seconds() > 0:
            ms = int(delta.total_seconds() * 1000)
            root.after(ms, self.schedule_write_attendance_to_db, root)

