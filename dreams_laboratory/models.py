from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    phone_number = models.CharField(max_length=15, unique=True, null=True, blank=True)
    is_phone_verified = models.BooleanField(default=False)


class People(models.Model):
  first_name = models.CharField(max_length=100)
  last_name = models.CharField(max_length=100)
  email = models.EmailField()
  profile_pic = models.URLField(max_length=300)
  bio = models.TextField(blank=True)
  role = models.ForeignKey('Role', on_delete=models.SET_NULL, null=True, blank=True)
  affiliation = models.CharField(max_length=200, blank=True, null=True)  # New field for affiliation

  def __str__(self):
      return f"{self.first_name} {self.last_name}"

# Other models remain unchanged

class Role(models.Model):
  position = models.CharField(max_length=100)
  start_date = models.DateField(default=timezone.now)
  stop_date = models.DateField(null=True, blank=True)
  email = models.EmailField()
  # Removed the affiliation field

  def __str__(self):
      return self.position
    


class Research(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    start_date = models.DateField()
    end_date = models.DateField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    effort = models.IntegerField(default=0, choices=[(i, i) for i in range(101)])
    style = models.CharField(max_length=200, default="'background-color: rgba(255,0,0,0.3);'")  # For example, could be 'background-color: rgba(255,0,0,0.3);'
    def __str__(self):
        return self.title





class Publication(models.Model):
  title = models.CharField(max_length=200)
  authors = models.TextField(blank=True, null=True)  # Changed to TextField in case there are many authors
  publication_date = models.DateField()
  abstract = models.TextField(blank=True, null=True)
  link = models.URLField(blank=True)
  doi = models.CharField(max_length=200, blank=True, null=True)

  def __str__(self):
      return self.title


class FundingSource(models.Model):
  source_name = models.CharField(max_length=255)
  photo_url = models.URLField(max_length=300, blank=True, null=True)  # Added photo URL field

  def __str__(self):
      return self.source_name

class Project(models.Model):
    title = models.CharField(max_length=200, default='Untitled Project')  # Added title field with default value
    funding_source = models.ForeignKey(FundingSource, on_delete=models.SET_NULL, null=True, blank=True, related_name='funded_projects')
    team_members = models.ManyToManyField(People, related_name='projects')
    publications = models.ManyToManyField(Publication, related_name='projects')
    research_areas = models.ManyToManyField(Research, related_name='projects')
    website_url = models.URLField(max_length=300, blank=True, null=True)  # Added URL field for project website

    def __str__(self):
        return self.title



class Asset(models.Model):
  asset_name = models.CharField(max_length=200,blank=True, null=True)
  description = models.TextField(blank=True, null=True)
  person = models.ForeignKey(People, on_delete=models.CASCADE, related_name='assets')
  project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='assets')

  def __str__(self):
    return self.asset_name

class Photo(models.Model):
  projects = models.ManyToManyField(Project, blank=True)
  researches = models.ManyToManyField(Research, blank=True)
  assets = models.ManyToManyField(Asset, blank=True)
  photo_url = models.URLField(max_length=300)
  caption = models.TextField(blank=True, default=' ')

  def __str__(self):
      return self.photo_url  # Or any other string representation you prefer.