from django.db import models
import uuid
from django.conf import settings

class Container(models.Model):
    SESSION_TYPES = [
        ('guest', 'Guest Session'),
        ('user', 'User Session'),
    ]

    container_id = models.CharField(max_length=64, unique=True)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=20)
    created = models.DateTimeField()
    ports = models.JSONField(default=dict)
    image = models.CharField(max_length=255)
    
    # Session information
    session_type = models.CharField(max_length=10, choices=SESSION_TYPES, default='guest')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='containers'
    )
    last_accessed = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['session_type', 'status']),
            models.Index(fields=['user', 'status']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.status})"
    
    def is_active(self):
        return self.status == 'running'
