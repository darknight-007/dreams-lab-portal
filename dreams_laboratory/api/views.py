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
            'local_position_odom': '/api/telemetry/local-position-odom/',
            'gps_fix_raw': '/api/telemetry/gps-fix-raw/',
            'gps_fix_estimated': '/api/telemetry/gps-fix-estimated/',
            'batch': '/api/telemetry/batch/'
        },
        'methods': {
            'local_position_odom': 'POST',
            'gps_fix_raw': 'POST',
            'gps_fix_estimated': 'POST',
            'batch': 'POST'
        },
        'documentation': 'See /api/telemetry/README.md or check the API documentation'
    })


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

