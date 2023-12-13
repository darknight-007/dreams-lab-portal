# deepgis/models.py

from django.contrib.gis.db import models

class TiledGISInfo(models.Model):
    name = models.CharField(max_length=200)
    geo_data = models.PolygonField()
    zoom_level = models.IntegerField()
    tile_data = models.BinaryField()  # Store tile binary data or use a FileField for file paths
