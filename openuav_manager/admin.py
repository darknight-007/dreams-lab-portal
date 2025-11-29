from django.contrib import admin
from .models import Container, RosbagFile, RosbagSession

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


@admin.register(RosbagFile)
class RosbagFileAdmin(admin.ModelAdmin):
    list_display = ('name', 'size_display', 'duration', 'topic_count', 'indexed')
    list_filter = ('indexed',)
    search_fields = ('name', 'path')
    readonly_fields = ('path', 'name', 'size', 'duration', 'topics', 'metadata', 'indexed')
    ordering = ('-indexed',)
    
    def size_display(self, obj):
        return obj.get_size_display()
    size_display.short_description = 'Size'
    
    def topic_count(self, obj):
        return len(obj.topics) if obj.topics else 0
    topic_count.short_description = 'Topics'


@admin.register(RosbagSession)
class RosbagSessionAdmin(admin.ModelAdmin):
    list_display = ('rosbag', 'visualization_type', 'container', 'user', 'is_playing', 'created')
    list_filter = ('visualization_type', 'is_playing', 'created')
    search_fields = ('rosbag__name', 'container__name', 'user__username')
    readonly_fields = ('created', 'last_accessed')
    ordering = ('-created',) 