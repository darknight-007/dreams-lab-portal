"""
RESTful API views for telemetry data endpoints.
Handles POST requests for GPSFixRaw, GPSFixEstimated, and LocalPositionOdom.
"""
import json
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.db import transaction
from django.conf import settings

from dreams_laboratory.models import (
    LocalPositionOdom, GPSFixRaw, GPSFixEstimated, DroneTelemetrySession
)
from .serializers import (
    LocalPositionOdomSerializer, GPSFixRawSerializer, GPSFixEstimatedSerializer
)
from django.db.models import Q


@csrf_exempt
@require_http_methods(["GET", "OPTIONS"])
def api_info(request):
    """
    GET endpoint for API information.
    Returns available endpoints and API version.
    """
    return JsonResponse({
        'api': 'Pixhawk Telemetry API',
        'version': '1.0',
        'endpoints': {
            'create_session': '/api/telemetry/session/create/',
            'local_position_odom': '/api/telemetry/local-position-odom/',
            'gps_fix_raw': '/api/telemetry/gps-fix-raw/',
            'gps_fix_estimated': '/api/telemetry/gps-fix-estimated/',
            'batch': '/api/telemetry/batch/'
        },
        'methods': {
            'create_session': 'POST',
            'local_position_odom': 'POST',
            'gps_fix_raw': 'POST',
            'gps_fix_estimated': 'POST',
            'batch': 'POST'
        },
        'documentation': 'See /api/telemetry/README.md or check the API documentation'
    })


@csrf_exempt
@require_http_methods(["POST"])
def create_telemetry_session(request):
    """
    POST endpoint to create a new telemetry session.
    
    Expected JSON payload:
    {
        "session_id": "test_session_20241123_120000_abc123",
        "asset_name": "RV Karin Valentine",  # Optional, will create if doesn't exist
        "project_title": "Tempe Town Lake Survey",  # Optional
        "flight_mode": "AUTO",  # Optional
        "mission_type": "Lake Survey",  # Optional
        "notes": "Test session"  # Optional
    }
    
    Returns:
        - 201 Created: Successfully created
        - 400 Bad Request: Validation errors
        - 500 Internal Server Error: Server error
    """
    data = parse_json_body(request)
    
    if data is None:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    
    # Validate required fields
    if 'session_id' not in data:
        return JsonResponse({
            'error': 'Validation failed',
            'errors': {'session_id': 'This field is required.'}
        }, status=400)
    
    session_id = data['session_id']
    
    try:
        with transaction.atomic():
            from dreams_laboratory.models import Asset, People, Project, DroneTelemetrySession
            from django.utils import timezone
            
            # Get or create asset
            asset_name = data.get('asset_name', 'RV Karin Valentine')
            asset = None
            
            try:
                asset = Asset.objects.get(asset_name=asset_name)
            except Asset.DoesNotExist:
                # Create asset if it doesn't exist
                person, _ = People.objects.get_or_create(
                    email='test@dreamslab.asu.edu',
                    defaults={
                        'first_name': 'Test',
                        'last_name': 'Operator',
                        'profile_pic': 'https://via.placeholder.com/150',
                        'bio': 'Test operator for telemetry data injection'
                    }
                )
                
                project, _ = Project.objects.get_or_create(
                    title=data.get('project_title', 'Tempe Town Lake Survey'),
                    defaults={
                        'website_url': 'https://dreamslab.asu.edu'
                    }
                )
                
                asset, _ = Asset.objects.get_or_create(
                    asset_name=asset_name,
                    defaults={
                        'description': 'Robotic boat for autonomous water surveys',
                        'person': person,
                        'project': project
                    }
                )
            
            # Get or create project if specified
            project = None
            if 'project_title' in data:
                project, _ = Project.objects.get_or_create(
                    title=data['project_title'],
                    defaults={
                        'website_url': 'https://dreamslab.asu.edu'
                    }
                )
            
            # Create or get session
            session, created = DroneTelemetrySession.objects.get_or_create(
                session_id=session_id,
                defaults={
                    'asset': asset,
                    'project': project,
                    'start_time': timezone.now(),
                    'flight_mode': data.get('flight_mode', 'AUTO'),
                    'mission_type': data.get('mission_type', 'Lake Survey'),
                    'notes': data.get('notes', f'Telemetry session created via API (Session ID: {session_id})')
                }
            )
            
            if not created:
                return JsonResponse({
                    'success': True,
                    'message': 'Session already exists',
                    'session_id': session.session_id,
                    'created': False,
                    'start_time': session.start_time.isoformat()
                }, status=200)
            
            return JsonResponse({
                'success': True,
                'message': 'Telemetry session created successfully',
                'session_id': session.session_id,
                'created': True,
                'asset': asset.asset_name,
                'project': project.title if project else None,
                'start_time': session.start_time.isoformat()
            }, status=201)
    
    except Exception as e:
        return JsonResponse({
            'error': 'Failed to create session',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)


def parse_json_body(request):
    """Parse JSON body from request"""
    try:
        if hasattr(request, 'body') and request.body:
            return json.loads(request.body.decode('utf-8'))
        return {}
    except json.JSONDecodeError:
        return None


@csrf_exempt
@require_http_methods(["POST"])
def post_local_position_odom(request):
    """
    POST endpoint for LocalPositionOdom telemetry data.
    
    Expected JSON payload:
    {
        "session_id": "flight_2024_11_23_001",
        "timestamp": "2024-11-23T01:00:00Z" or unix timestamp,
        "timestamp_usec": 1234567890,
        "x": 10.5,
        "y": 5.2,
        "z": -2.1,
        "vx": 1.2,
        "vy": 0.8,
        "vz": -0.1,
        "heading": 0.785,
        "xy_valid": true,
        "z_valid": true,
        ...
    }
    
    Returns:
        - 201 Created: Successfully created
        - 400 Bad Request: Validation errors
        - 500 Internal Server Error: Server error
    """
    data = parse_json_body(request)
    
    if data is None:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    
    # Validate and parse data
    try:
        parsed_data, errors = LocalPositionOdomSerializer.validate_and_parse(data)
    except Exception as e:
        return JsonResponse({
            'error': 'Validation error',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)
    
    if errors or parsed_data is None:
        return JsonResponse({
            'error': 'Validation failed',
            'errors': errors or {'general': 'Failed to parse data'}
        }, status=400)
    
    try:
        # Create the record
        with transaction.atomic():
            # Defensive cleanup: ensure session_id is removed (should already be done by serializer)
            parsed_data.pop('session_id', None)
            odom = LocalPositionOdom.objects.create(**parsed_data)
            
            # Update session statistics
            session = parsed_data['session']
            session.total_telemetry_points += 1
            session.save(update_fields=['total_telemetry_points'])
        
        return JsonResponse({
            'success': True,
            'id': odom.id,
            'message': 'Local position odometry data created successfully',
            'timestamp': odom.timestamp.isoformat()
        }, status=201)
    
    except Exception as e:
        import traceback
        from django.conf import settings
        return JsonResponse({
            'error': 'Failed to create record',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def post_gps_fix_raw(request):
    """
    POST endpoint for GPSFixRaw telemetry data.
    
    Expected JSON payload:
    {
        "session_id": "flight_2024_11_23_001",
        "timestamp": "2024-11-23T01:00:00Z",
        "timestamp_usec": 1234567891,
        "latitude": 33.4255000,
        "longitude": -111.9400000,
        "altitude": 352.5,
        "fix_type": 3,
        "satellites_visible": 12,
        "hdop": 1.2,
        "eph": 2.5,
        ...
    }
    
    Returns:
        - 201 Created: Successfully created
        - 400 Bad Request: Validation errors
        - 500 Internal Server Error: Server error
    """
    data = parse_json_body(request)
    
    if data is None:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    
    # Validate and parse data
    try:
        parsed_data, errors = GPSFixRawSerializer.validate_and_parse(data)
    except Exception as e:
        return JsonResponse({
            'error': 'Validation error',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)
    
    if errors or parsed_data is None:
        return JsonResponse({
            'error': 'Validation failed',
            'errors': errors or {'general': 'Failed to parse data'}
        }, status=400)
    
    try:
        # Create the record
        with transaction.atomic():
            # Defensive cleanup: ensure session_id is removed (should already be done by serializer)
            parsed_data.pop('session_id', None)
            gps_raw = GPSFixRaw.objects.create(**parsed_data)
            
            # Update session statistics
            session = parsed_data['session']
            session.total_telemetry_points += 1
            session.save(update_fields=['total_telemetry_points'])
        
        return JsonResponse({
            'success': True,
            'id': gps_raw.id,
            'message': 'Raw GPS fix data created successfully',
            'timestamp': gps_raw.timestamp.isoformat()
        }, status=201)
    
    except Exception as e:
        return JsonResponse({
            'error': 'Failed to create record',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def post_gps_fix_estimated(request):
    """
    POST endpoint for GPSFixEstimated telemetry data.
    
    Expected JSON payload:
    {
        "session_id": "flight_2024_11_23_001",
        "timestamp": "2024-11-23T01:00:00Z",
        "timestamp_usec": 1234567892,
        "latitude": 33.4255010,
        "longitude": -111.9400010,
        "altitude": 352.3,
        "vel_n_m_s": 1.18,
        "vel_e_m_s": 0.82,
        "estimator_type": "EKF2",
        "position_valid": true,
        "eph": 1.8,
        ...
    }
    
    Returns:
        - 201 Created: Successfully created
        - 400 Bad Request: Validation errors
        - 500 Internal Server Error: Server error
    """
    data = parse_json_body(request)
    
    if data is None:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    
    # Validate and parse data
    try:
        parsed_data, errors = GPSFixEstimatedSerializer.validate_and_parse(data)
    except Exception as e:
        return JsonResponse({
            'error': 'Validation error',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)
    
    if errors or parsed_data is None:
        return JsonResponse({
            'error': 'Validation failed',
            'errors': errors or {'general': 'Failed to parse data'}
        }, status=400)
    
    try:
        # Create the record
        with transaction.atomic():
            # Defensive cleanup: ensure session_id is removed (should already be done by serializer)
            parsed_data.pop('session_id', None)
            gps_est = GPSFixEstimated.objects.create(**parsed_data)
            
            # Update session statistics
            session = parsed_data['session']
            session.total_telemetry_points += 1
            session.save(update_fields=['total_telemetry_points'])
        
        return JsonResponse({
            'success': True,
            'id': gps_est.id,
            'message': 'Estimated GPS fix data created successfully',
            'timestamp': gps_est.timestamp.isoformat()
        }, status=201)
    
    except Exception as e:
        return JsonResponse({
            'error': 'Failed to create record',
            'message': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def post_telemetry_batch(request):
    """
    POST endpoint for batch telemetry data (multiple records at once).
    
    Expected JSON payload:
    {
        "local_position_odom": [
            { "session_id": "...", "x": 10.5, "y": 5.2, ... },
            { "session_id": "...", "x": 10.6, "y": 5.3, ... }
        ],
        "gps_fix_raw": [
            { "session_id": "...", "latitude": 33.4255, ... },
            ...
        ],
        "gps_fix_estimated": [
            { "session_id": "...", "latitude": 33.4255, ... },
            ...
        ]
    }
    
    Returns:
        - 201 Created: Successfully created (with counts)
        - 400 Bad Request: Validation errors
        - 500 Internal Server Error: Server error
    """
    data = parse_json_body(request)
    
    if data is None:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    
    results = {
        'local_position_odom': {'created': 0, 'errors': []},
        'gps_fix_raw': {'created': 0, 'errors': []},
        'gps_fix_estimated': {'created': 0, 'errors': []}
    }
    
    try:
        with transaction.atomic():
            # Process LocalPositionOdom batch
            if 'local_position_odom' in data and isinstance(data['local_position_odom'], list):
                for idx, item in enumerate(data['local_position_odom']):
                    parsed_data, errors = LocalPositionOdomSerializer.validate_and_parse(item)
                    if errors:
                        results['local_position_odom']['errors'].append({
                            'index': idx,
                            'errors': errors
                        })
                    else:
                        try:
                            # Defensive cleanup: ensure session_id is removed
                            parsed_data.pop('session_id', None)
                            LocalPositionOdom.objects.create(**parsed_data)
                            results['local_position_odom']['created'] += 1
                            
                            # Update session statistics
                            session = parsed_data['session']
                            session.total_telemetry_points += 1
                            session.save(update_fields=['total_telemetry_points'])
                        except Exception as e:
                            results['local_position_odom']['errors'].append({
                                'index': idx,
                                'error': str(e)
                            })
            
            # Process GPSFixRaw batch
            if 'gps_fix_raw' in data and isinstance(data['gps_fix_raw'], list):
                for idx, item in enumerate(data['gps_fix_raw']):
                    parsed_data, errors = GPSFixRawSerializer.validate_and_parse(item)
                    if errors:
                        results['gps_fix_raw']['errors'].append({
                            'index': idx,
                            'errors': errors
                        })
                    else:
                        try:
                            # Defensive cleanup: ensure session_id is removed
                            parsed_data.pop('session_id', None)
                            GPSFixRaw.objects.create(**parsed_data)
                            results['gps_fix_raw']['created'] += 1
                            
                            # Update session statistics
                            session = parsed_data['session']
                            session.total_telemetry_points += 1
                            session.save(update_fields=['total_telemetry_points'])
                        except Exception as e:
                            results['gps_fix_raw']['errors'].append({
                                'index': idx,
                                'error': str(e)
                            })
            
            # Process GPSFixEstimated batch
            if 'gps_fix_estimated' in data and isinstance(data['gps_fix_estimated'], list):
                for idx, item in enumerate(data['gps_fix_estimated']):
                    parsed_data, errors = GPSFixEstimatedSerializer.validate_and_parse(item)
                    if errors:
                        results['gps_fix_estimated']['errors'].append({
                            'index': idx,
                            'errors': errors
                        })
                    else:
                        try:
                            # Defensive cleanup: ensure session_id is removed
                            parsed_data.pop('session_id', None)
                            GPSFixEstimated.objects.create(**parsed_data)
                            results['gps_fix_estimated']['created'] += 1
                            
                            # Update session statistics
                            session = parsed_data['session']
                            session.total_telemetry_points += 1
                            session.save(update_fields=['total_telemetry_points'])
                        except Exception as e:
                            results['gps_fix_estimated']['errors'].append({
                                'index': idx,
                                'error': str(e)
                            })
        
        total_created = (results['local_position_odom']['created'] + 
                        results['gps_fix_raw']['created'] + 
                        results['gps_fix_estimated']['created'])
        
        has_errors = any(len(v['errors']) > 0 for v in results.values())
        
        status_code = 201 if not has_errors else 207  # 207 Multi-Status for partial success
        
        return JsonResponse({
            'success': True,
            'message': f'Batch processed: {total_created} records created',
            'results': results
        }, status=status_code)
    
    except Exception as e:
        return JsonResponse({
            'error': 'Failed to process batch',
            'message': str(e)
        }, status=500)


def gps_to_geojson(gps_points, include_properties=True):
    """
    Convert a list of GPSFixRaw objects to GeoJSON format.
    
    Args:
        gps_points: QuerySet or list of GPSFixRaw objects, ordered by timestamp
        include_properties: Whether to include additional properties in features
    
    Returns:
        dict: GeoJSON FeatureCollection
    """
    features = []
    coordinates = []
    
    for point in gps_points:
        # GeoJSON uses [longitude, latitude, altitude] format
        coord = [
            float(point.longitude),
            float(point.latitude),
            float(point.altitude) if point.altitude else 0.0
        ]
        coordinates.append(coord)
        
        if include_properties:
            # Create a point feature for each GPS fix
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': coord
                },
                'properties': {
                    'timestamp': point.timestamp.isoformat(),
                    'altitude': float(point.altitude) if point.altitude else None,
                    'fix_type': point.fix_type,
                    'satellites_visible': point.satellites_visible,
                    'satellites_used': point.satellites_used,
                    'eph': float(point.eph) if point.eph else None,
                    'epv': float(point.epv) if point.epv else None,
                    'hdop': float(point.hdop) if point.hdop else None,
                    'vel_m_s': float(point.vel_m_s) if point.vel_m_s else None,
                    'jamming_state': point.jamming_state,
                }
            }
            features.append(feature)
    
    # Create LineString for the path
    if len(coordinates) > 1:
        linestring_feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coordinates
            },
            'properties': {
                'point_count': len(coordinates),
                'type': 'path'
            }
        }
        features.insert(0, linestring_feature)  # Add path as first feature
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


@csrf_exempt
@require_http_methods(["GET", "OPTIONS"])
def get_session_path(request, session_id=None):
    """
    GET endpoint to retrieve GPS path for a telemetry session.
    
    URL: /api/telemetry/sessions/<session_id>/path/
    
    Query parameters:
        - format: 'geojson' (default) or 'points'
        - include_properties: 'true' (default) or 'false'
    
    Returns:
        - 200 OK: GeoJSON FeatureCollection with session path
        - 404 Not Found: Session not found
        - 400 Bad Request: Invalid parameters
    """
    if not session_id:
        return JsonResponse({
            'error': 'Session ID required'
        }, status=400)
    
    try:
        session = DroneTelemetrySession.objects.get(session_id=session_id)
    except DroneTelemetrySession.DoesNotExist:
        return JsonResponse({
            'error': f'Session "{session_id}" not found'
        }, status=404)
    
    # Get query parameters
    format_type = request.GET.get('format', 'geojson').lower()
    include_props = request.GET.get('include_properties', 'true').lower() == 'true'
    
    # Get GPS points for this session, ordered by timestamp
    gps_points = GPSFixRaw.objects.filter(session=session).order_by('timestamp')
    
    if not gps_points.exists():
        return JsonResponse({
            'error': f'No GPS data found for session "{session_id}"'
        }, status=404)
    
    # Format response based on requested format
    if format_type == 'geojson':
        geojson_data = gps_to_geojson(gps_points, include_properties=include_props)
        
        # Add session metadata
        response_data = {
            'session_id': session.session_id,
            'session_info': {
                'asset': session.asset.asset_name if session.asset else None,
                'project': session.project.title if session.project else None,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'flight_mode': session.flight_mode,
                'mission_type': session.mission_type,
                'total_points': gps_points.count()
            },
            'geojson': geojson_data
        }
        
        return JsonResponse(response_data, status=200)
    
    elif format_type == 'points':
        # Return simple array of points
        points = []
        for point in gps_points:
            points.append({
                'timestamp': point.timestamp.isoformat(),
                'longitude': float(point.longitude),
                'latitude': float(point.latitude),
                'altitude': float(point.altitude) if point.altitude else None,
                'fix_type': point.fix_type
            })
        
        return JsonResponse({
            'session_id': session.session_id,
            'points': points,
            'count': len(points)
        }, status=200)
    
    else:
        return JsonResponse({
            'error': f'Invalid format: {format_type}. Use "geojson" or "points"'
        }, status=400)


@csrf_exempt
@require_http_methods(["GET", "OPTIONS"])
def list_sessions(request):
    """
    GET endpoint to list all telemetry sessions with metadata.
    
    URL: /api/telemetry/sessions/
    
    Query parameters:
        - asset_name: Filter by asset name
        - project_title: Filter by project title
        - has_gps: 'true' to only show sessions with GPS data
    
    Returns:
        - 200 OK: List of sessions with metadata
    """
    sessions = DroneTelemetrySession.objects.all().select_related('asset', 'project')
    
    # Apply filters
    asset_name = request.GET.get('asset_name')
    if asset_name:
        sessions = sessions.filter(asset__asset_name__icontains=asset_name)
    
    project_title = request.GET.get('project_title')
    if project_title:
        sessions = sessions.filter(project__title__icontains=project_title)
    
    has_gps = request.GET.get('has_gps', 'false').lower() == 'true'
    if has_gps:
        # Only include sessions that have GPS data
        sessions = sessions.filter(gps_fixes_raw__isnull=False).distinct()
    
    # Order by start_time (most recent first)
    sessions = sessions.order_by('-start_time')
    
    # Build response
    session_list = []
    for session in sessions:
        gps_count = GPSFixRaw.objects.filter(session=session).count()
        
        session_data = {
            'session_id': session.session_id,
            'asset': session.asset.asset_name if session.asset else None,
            'project': session.project.title if session.project else None,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'flight_mode': session.flight_mode,
            'mission_type': session.mission_type,
            'gps_point_count': gps_count,
            'has_gps_data': gps_count > 0,
            'path_url': f'/api/telemetry/sessions/{session.session_id}/path/'
        }
        session_list.append(session_data)
    
    return JsonResponse({
        'sessions': session_list,
        'count': len(session_list)
    }, status=200)

