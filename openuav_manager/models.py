from django.db import models

class Container(models.Model):
    container_id = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=20)
    created = models.DateTimeField()
    ports = models.JSONField(default=dict)
    image = models.CharField(max_length=255)
    
    def __str__(self):
        return f"{self.name} ({self.status})"
