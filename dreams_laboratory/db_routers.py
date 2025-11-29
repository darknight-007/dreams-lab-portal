"""
Database Router for World Sampler Models

Routes World Sampler models to the deepgis-xr database
while keeping all other models in the default database.
"""


class WorldSamplerRouter:
    """
    A router to control database operations for World Sampler models.
    """
    
    route_app_labels = {'world_sampler'}
    deepgis_xr_models = {
        'SampledLocation', 'SamplingSession', 'DistributionUpdate',
        'Mission', 'MissionWaypoint', 'Vehicle', 'VehicleType'
    }
    
    def db_for_read(self, model, **hints):
        """
        Route read operations for World Sampler models to deepgis_xr database.
        """
        if model.__name__ in self.deepgis_xr_models:
            return 'deepgis_xr'
        return None
    
    def db_for_write(self, model, **hints):
        """
        Route write operations for World Sampler models to deepgis_xr database.
        """
        if model.__name__ in self.deepgis_xr_models:
            return 'deepgis_xr'
        return None
    
    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if both models are in the same database.
        """
        if (obj1.__class__.__name__ in self.deepgis_xr_models and
            obj2.__class__.__name__ in self.deepgis_xr_models):
            return True
        return None
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Ensure World Sampler models only appear in deepgis_xr database.
        """
        if model_name in [m.lower() for m in self.deepgis_xr_models]:
            return db == 'deepgis_xr'
        return None

