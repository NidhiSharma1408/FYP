from django.db import models
import numpy as np

class AttendanceModel(models.Model):
    name = models.CharField(max_length=100)
    timestamp = models.DateTimeField()

    def __str__(self):
        return self.name 

class FaceModel(models.Model):
    name = models.CharField(max_length=255)
    roll_no = models.CharField(max_length=30, blank=True)
    embedding = models.BinaryField()
    def set_face_embedding(self, embedding):
        # Convert numpy array to bytes before storing in BinaryField
        self.embedding = embedding.tobytes()

    def get_face_embedding(self):
        # Convert bytes to numpy array when retrieving from BinaryField
        return np.frombuffer(self.embedding, dtype=np.float32)
    def __str__(self):
        return self.name