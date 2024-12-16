from django.contrib import admin
from .models import People, Research, Publication, Project, Role, Photo, Asset, FundingSource
from .models import Photo
from django.utils.html import format_html
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    pass


# Register your models here.
admin.site.register(People)
admin.site.register(Research)
admin.site.register(Publication)
admin.site.register(Project)

admin.site.register(Role)
@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ('thumbnail', 'photo_url',)  # Add other fields to display if needed

    def thumbnail(self, obj):
        return format_html('<img src="{}" style="width: 50px; height:auto;">', obj.photo_url)
    
    thumbnail.short_description = 'Thumbnail'  # Optional: Sets column name
  
@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('asset_name',)  # Add other fields if you want them to show up in the list as well


@admin.register(FundingSource)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('source_name',)  # Add other fields if you want them to show up in the list as well


