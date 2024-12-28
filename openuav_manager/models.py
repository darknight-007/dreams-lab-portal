from django.db import models
from django.conf import settings

class Container(models.Model):
    container_id = models.CharField(max_length=64, unique=True)
    unique_id = models.CharField(max_length=36)  # UUID
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=20)
    created = models.DateTimeField(auto_now_add=True)
    ports = models.JSONField(default=dict)
    image = models.CharField(max_length=100)
    session_type = models.CharField(max_length=20, default='guest')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    subdomain = models.CharField(max_length=100, unique=True, null=True, blank=True)
    container_ip = models.CharField(max_length=50, blank=True, null=True)
    network = models.CharField(max_length=50, default='openuav_network')

    def __str__(self):
        return f"{self.name} ({self.status})"

    def get_novnc_url(self):
        if self.subdomain:
            return f"https://{self.subdomain}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
        return None

    def get_vnc_address(self):
        if self.subdomain:
            return f"{self.subdomain}.deepgis.org:5901"
        return None

    def update_container_ip(self):
        """Update container IP address from Docker"""
        from django.core.management import call_command
        call_command('update_container_ip', self.container_id)

    class Meta:
        indexes = [
            models.Index(fields=['subdomain']),
            models.Index(fields=['container_ip']),
            models.Index(fields=['status'])
        ]
