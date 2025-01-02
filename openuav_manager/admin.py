from django.contrib import admin
from .models import Container

@admin.register(Container)
class ContainerAdmin(admin.ModelAdmin):
    list_display = ('name', 'container_id', 'short_id', 'status', 'created', 'image')
    list_filter = ('status', 'image')
    search_fields = ('name', 'container_id', 'short_id')
    readonly_fields = ('container_id', 'short_id', 'created')
    ordering = ('-created',)

    def has_add_permission(self, request):
        # Containers should only be created through the OpenUAV interface
        return False 