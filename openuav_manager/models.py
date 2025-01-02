from django.db import models
import uuid
from django.conf import settings

class Container(models.Model):
    SESSION_TYPES = [
        ('guest', 'Guest Session'),
        ('user', 'User Session'),
    ]

    # Container identification
    container_id = models.CharField(max_length=64, unique=True, help_text='Full container ID')
    short_id = models.CharField(max_length=12, help_text='Short container ID (first 12 characters)')
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=20)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    ports = models.JSONField(default=dict, help_text='Mapping of host ports to container ports')
    image = models.CharField(max_length=255)
    
    # Session information
    session_type = models.CharField(max_length=10, choices=SESSION_TYPES, default='guest')
    session_id = models.CharField(max_length=64, null=True, blank=True, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='containers'
    )
    last_accessed = models.DateTimeField(auto_now=True)
    vnc_url = models.URLField(max_length=255, blank=True, null=True, help_text='URL for VNC access')
    
    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['session_type', 'status']),
            models.Index(fields=['user', 'status']),
            models.Index(fields=['session_id', 'status']),
            models.Index(fields=['short_id']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.short_id})"
    
    def save(self, *args, **kwargs):
        # Set short_id from container_id if not already set
        if self.container_id and not self.short_id:
            self.short_id = self.container_id[:12]
        super().save(*args, **kwargs)
    
    def is_active(self):
        return self.status == 'running'
    
    def get_vnc_url(self):
        """Get the VNC URL for this container"""
        if not self.is_active():
            return None
        
        # Extract container ID from name (format: openuav-xxxxxxxx)
        container_id = self.name.split('-')[1] if '-' in self.name else None
        if not container_id:
            return None
        
        # Use digital-twin subdomain format
        return f"https://digital-twin-{container_id}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"

    def get_port_mappings(self):
        """Get the port mappings for this container"""
        return {
            'vnc': self.ports.get('5901', '5901'),
            'novnc': self.ports.get('6080', '6080')
        }
