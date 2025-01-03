from django.contrib import admin
from .models import People, Research, Publication, Project, Role, Photo, Asset, FundingSource, QuizSubmission
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

@admin.register(QuizSubmission)
class QuizSubmissionAdmin(admin.ModelAdmin):
    list_display = ('get_submission_title', 'submission_date', 'total_score')
    list_filter = ('submission_date', 'quiz_id')
    readonly_fields = ('quiz_id', 'submission_date', 'total_score', 
                      'cv_score', 'slam_score', 'estimation_score', 
                      'sensing_score', 'motion_score', 'neural_score',
                      'q1', 'q2', 'q3', 'q4', 'q5', 
                      'q6', 'q7', 'q8', 'q9', 'q10',
                      'q11', 'q12', 'q13', 'q14', 'q15')
    
    def get_submission_title(self, obj):
        return f"{obj.email} - {obj.submission_date.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    get_submission_title.short_description = 'Submission Info'
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return True


