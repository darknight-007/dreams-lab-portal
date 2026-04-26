"""kernelcal_earth_rover: edge pipeline for Dreams Lab earth_rover platforms.

Sensor messages -> superquadric primitives -> attributed superquadrics
-> packed binary payloads -> deepgis-xr POST /api/v1/observe.

Three nodes:

* :mod:`kernelcal_earth_rover.sq_fit_node` -- consumes
  ``sensor_msgs/PointCloud2`` and emits a *cluster batch* of fitted
  :class:`kernelcal.distinction_game.geometry.Superquadric`.
* :mod:`kernelcal_earth_rover.attributor_node` -- joins the SQ stream
  with the rover's MicaSense / OceanOptics / LiDAR-intensity feeds and
  populates the geometry, property, and spectrum trailers.
* :mod:`kernelcal_earth_rover.observation_publisher_node` -- batches
  attributed SQs into the kernelcal codec wire format and POSTs them
  to deepgis-xr at the configured rate.

The package is intentionally split so that the SQ fitter (CPU- and
LiDAR-bound) can run on a different ROS executor than the attributor
(IO-bound, multispectral image arrival) and the publisher (network).
"""

__version__ = "0.1.0"
