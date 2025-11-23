#!/usr/bin/env python3
"""
Script to inject 100 test telemetry data points via REST API.
Uses the API endpoints to post data, simulating real-world usage.

This script is standalone and does NOT require Django - only needs the 'requests' library.

Usage:
    python scripts/inject_telemetry_via_api.py [--url API_URL] [--session-id SESSION_ID]

Configuration:
    Set API_BASE_URL below or pass --url argument
    Provide --session-id or it will be auto-generated (requires session to exist in DB)
"""

import sys
from datetime import datetime, timedelta
import math
import random
import requests
import json
import argparse

# ============================================================================
# CONFIGURATION - Set your API base URL here
# ============================================================================
# Default API base URL (can be overridden via command-line argument)
API_BASE_URL = "https://deepgis.org/api/telemetry"

# Alternative URLs (uncomment to use):
# API_BASE_URL = "http://localhost:8000/api/telemetry"  # Local development
# API_BASE_URL = "https://deepgis.org/api/telemetry"   # Production HTTPS
# API_BASE_URL = "http://172.20.0.10/api/telemetry"    # Docker internal network


def get_or_create_session_via_django(session_id=None):
    """
    Optional: Use Django to create session if Django is available.
    Returns session_id. If Django not available, returns provided session_id or generates one.
    """
    try:
        import os
        from pathlib import Path
        import django
        
        # Setup Django environment
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(BASE_DIR))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dreams_laboratory.settings')
        django.setup()
        
        from django.utils import timezone
        from dreams_laboratory.models import Asset, People, Project, DroneTelemetrySession
        
        # Get or create a test person
        person, _ = People.objects.get_or_create(
            email='test@dreamslab.asu.edu',
            defaults={
                'first_name': 'Test',
                'last_name': 'Operator',
                'profile_pic': 'https://via.placeholder.com/150',
                'bio': 'Test operator for telemetry data injection'
            }
        )
        
        # Get or create a test project
        project, _ = Project.objects.get_or_create(
            title='Tempe Town Lake Survey',
            defaults={
                'website_url': 'https://dreamslab.asu.edu'
            }
        )
        
        # Get or create the robotic boat asset
        asset, _ = Asset.objects.get_or_create(
            asset_name='RV Karin Valentine',
            defaults={
                'description': 'Robotic boat for autonomous water surveys',
                'person': person,
                'project': project
            }
        )
        
        # Update asset if needed
        if not asset.person or not asset.project:
            asset.person = person
            asset.project = project
            asset.save()
        
        # Create or get telemetry session
        if not session_id:
            session_id = f"tempe_town_lake_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        
        session, created = DroneTelemetrySession.objects.get_or_create(
            session_id=session_id,
            defaults={
                'asset': asset,
                'project': project,
                'start_time': timezone.now(),
                'flight_mode': 'AUTO',
                'mission_type': 'Lake Survey',
                'notes': 'Test data injection via API: 100 points along 100m path in Tempe Town Lake, Arizona'
            }
        )
        
        return session_id, session, True  # True = Django available
        
    except (ImportError, ModuleNotFoundError):
        # Django not available - return session_id only
        if not session_id:
            session_id = f"tempe_town_lake_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return session_id, None, False  # False = Django not available


def generate_path_points(num_points=100, path_length=100.0):
    """Generate points along a 100m path in Tempe Town Lake"""
    start_lat = 33.4255000
    start_lon = -111.9405000
    start_alt = 350.0
    
    lat_meters_per_degree = 111000.0
    lon_meters_per_degree = 111000.0 * math.cos(math.radians(start_lat))
    
    points = []
    base_time = datetime.utcnow()
    
    for i in range(num_points):
        progress = i / (num_points - 1) if num_points > 1 else 0
        distance_east = path_length * progress
        distance_north = 5.0 * math.sin(progress * math.pi * 2)
        
        lat_offset = distance_north / lat_meters_per_degree
        lon_offset = distance_east / lon_meters_per_degree
        
        lat = start_lat + lat_offset
        lon = start_lon + lon_offset
        
        lat_noise = random.uniform(-0.5, 0.5) / lat_meters_per_degree
        lon_noise = random.uniform(-0.5, 0.5) / lon_meters_per_degree
        
        lat += lat_noise
        lon += lon_noise
        
        x = distance_north + random.uniform(-0.1, 0.1)
        y = distance_east + random.uniform(-0.1, 0.1)
        z = -0.5 + random.uniform(-0.2, 0.2)
        
        speed = 1.0 + random.uniform(-0.2, 0.2)
        heading = math.atan2(distance_east, distance_north) if i > 0 else 0.0
        
        vx = speed * math.cos(heading) + random.uniform(-0.1, 0.1)
        vy = speed * math.sin(heading) + random.uniform(-0.1, 0.1)
        vz = random.uniform(-0.05, 0.05)
        
        timestamp = base_time + timedelta(seconds=i)
        timestamp_usec = int(timestamp.timestamp() * 1e6)
        
        points.append({
            'index': i,
            'timestamp': timestamp.isoformat() + 'Z',
            'timestamp_usec': timestamp_usec,
            'lat': lat,
            'lon': lon,
            'alt': start_alt + random.uniform(-0.1, 0.1),
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'heading': heading,
            'speed': speed,
            'ref_lat': start_lat,
            'ref_lon': start_lon,
            'ref_alt': start_alt,
        })
    
    return points


def post_telemetry_via_api(base_url, session_id, points):
    """Post telemetry data via REST API"""
    results = {
        'local_odom': {'success': 0, 'errors': []},
        'gps_raw': {'success': 0, 'errors': []},
        'gps_estimated': {'success': 0, 'errors': []}
    }
    
    # Store IDs for linking
    gps_raw_ids = []
    local_odom_ids = []
    
    print(f"\nPosting telemetry data via API for {len(points)} points...")
    
    for i, point in enumerate(points):
        if (i + 1) % 10 == 0:
            print(f"  Processing point {i + 1}/{len(points)}...")
        
        # 1. Post LocalPositionOdom
        local_odom_data = {
            'session_id': session_id,
            'timestamp': point['timestamp'],
            'timestamp_usec': point['timestamp_usec'],
            'x': point['x'],
            'y': point['y'],
            'z': point['z'],
            'vx': point['vx'],
            'vy': point['vy'],
            'vz': point['vz'],
            'heading': point['heading'],
            'heading_rate': random.uniform(-0.05, 0.05),
            'xy_valid': True,
            'z_valid': True,
            'v_xy_valid': True,
            'v_z_valid': True,
            'heading_valid': True,
            'position_covariance': [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.02],
            'velocity_covariance': [0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.1],
            'ref_lat': point['ref_lat'],
            'ref_lon': point['ref_lon'],
            'ref_alt': point['ref_alt'],
            'eph': random.uniform(0.3, 0.8),
            'epv': random.uniform(0.5, 1.2),
            'evh': random.uniform(0.1, 0.3),
            'evv': random.uniform(0.15, 0.4)
        }
        
        try:
            response = requests.post(
                f"{base_url}/local-position-odom/",
                json=local_odom_data,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            if response.status_code == 201:
                result = response.json()
                local_odom_ids.append(result.get('id'))
                results['local_odom']['success'] += 1
            else:
                results['local_odom']['errors'].append({
                    'index': i,
                    'status': response.status_code,
                    'error': response.json() if response.content else 'Unknown error'
                })
        except Exception as e:
            results['local_odom']['errors'].append({
                'index': i,
                'error': str(e)
            })
        
        # 2. Post GPSFixRaw
        gps_noise_lat = random.uniform(-2.0, 2.0) / 111000.0
        gps_noise_lon = random.uniform(-2.0, 2.0) / (111000.0 * math.cos(math.radians(point['lat'])))
        
        gps_raw_data = {
            'session_id': session_id,
            'timestamp': point['timestamp'],
            'timestamp_usec': point['timestamp_usec'],
            'latitude': point['lat'] + gps_noise_lat,
            'longitude': point['lon'] + gps_noise_lon,
            'altitude': point['alt'] + random.uniform(-1.0, 1.0),
            'fix_type': 3,
            'satellites_visible': random.randint(8, 14),
            'satellites_used': random.randint(7, 12),
            'hdop': random.uniform(0.8, 2.0),
            'vdop': random.uniform(1.0, 2.5),
            'pdop': random.uniform(1.5, 3.0),
            'eph': random.uniform(1.5, 3.5),
            'epv': random.uniform(2.0, 4.0),
            's_variance_m_s': random.uniform(0.3, 0.8),
            'vel_n_m_s': point['vx'] + random.uniform(-0.2, 0.2),
            'vel_e_m_s': point['vy'] + random.uniform(-0.2, 0.2),
            'vel_d_m_s': point['vz'] + random.uniform(-0.1, 0.1),
            'vel_m_s': point['speed'] + random.uniform(-0.2, 0.2),
            'cog_rad': point['heading'] + random.uniform(-0.1, 0.1),
            'time_utc_usec': point['timestamp_usec'],
            'noise_per_ms': random.randint(20, 40),
            'jamming_indicator': random.randint(0, 10),
            'jamming_state': 1,
            'device_id': 1
        }
        
        try:
            response = requests.post(
                f"{base_url}/gps-fix-raw/",
                json=gps_raw_data,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            if response.status_code == 201:
                result = response.json()
                gps_raw_ids.append(result.get('id'))
                results['gps_raw']['success'] += 1
            else:
                results['gps_raw']['errors'].append({
                    'index': i,
                    'status': response.status_code,
                    'error': response.json() if response.content else 'Unknown error'
                })
        except Exception as e:
            results['gps_raw']['errors'].append({
                'index': i,
                'error': str(e)
            })
        
        # 3. Post GPSFixEstimated (if we have IDs from previous posts)
        if len(gps_raw_ids) > 0 and len(local_odom_ids) > 0:
            est_noise_lat = random.uniform(-0.5, 0.5) / 111000.0
            est_noise_lon = random.uniform(-0.5, 0.5) / (111000.0 * math.cos(math.radians(point['lat'])))
            
            gps_est_data = {
                'session_id': session_id,
                'timestamp': point['timestamp'],
                'timestamp_usec': point['timestamp_usec'],
                'latitude': point['lat'] + est_noise_lat,
                'longitude': point['lon'] + est_noise_lon,
                'altitude': point['alt'] + random.uniform(-0.3, 0.3),
                'vel_n_m_s': point['vx'] + random.uniform(-0.1, 0.1),
                'vel_e_m_s': point['vy'] + random.uniform(-0.1, 0.1),
                'vel_d_m_s': point['vz'] + random.uniform(-0.05, 0.05),
                'position_covariance': [0.3, 0, 0, 0, 0.3, 0, 0, 0, 0.8],
                'velocity_covariance': [0.08, 0, 0, 0, 0.08, 0, 0, 0, 0.15],
                'eph': random.uniform(0.8, 1.8),
                'epv': random.uniform(1.2, 2.2),
                'evh': random.uniform(0.08, 0.2),
                'evv': random.uniform(0.12, 0.3),
                'estimator_type': 'EKF2',
                'confidence': random.uniform(0.85, 0.98),
                'position_valid': True,
                'velocity_valid': True,
                'raw_gps_fix_id': gps_raw_ids[-1] if gps_raw_ids else None,
                'local_position_id': local_odom_ids[-1] if local_odom_ids else None
            }
            
            try:
                response = requests.post(
                    f"{base_url}/gps-fix-estimated/",
                    json=gps_est_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=5.0
                )
                if response.status_code == 201:
                    results['gps_estimated']['success'] += 1
                else:
                    results['gps_estimated']['errors'].append({
                        'index': i,
                        'status': response.status_code,
                        'error': response.json() if response.content else 'Unknown error'
                    })
            except Exception as e:
                results['gps_estimated']['errors'].append({
                    'index': i,
                    'error': str(e)
                })
    
    return results


def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Inject telemetry test data via REST API (standalone - no Django required)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python {sys.argv[0]}                                    # Use default URL, auto-generate session
  python {sys.argv[0]} --url http://localhost:8000/api/telemetry
  python {sys.argv[0]} --url https://deepgis.org/api/telemetry
  python {sys.argv[0]} --session-id my_existing_session  # Use existing session
        """
    )
    parser.add_argument(
        '--url',
        type=str,
        default=API_BASE_URL,
        help=f'API base URL (default: {API_BASE_URL})'
    )
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Session ID to use (must exist in database). If not provided, will try to create via Django or auto-generate.'
    )
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')  # Remove trailing slash if present
    
    print("=" * 70)
    print("Tempe Town Lake Telemetry Data Injection (via API)")
    print("Robotic Boat: RV Karin Valentine")
    print("=" * 70)
    
    # Setup session (try Django if available, otherwise use provided/generated session_id)
    print("\n1. Setting up session...")
    session_id, session, django_available = get_or_create_session_via_django(args.session_id)
    
    if django_available:
        print(f"   ✓ Django available - Session created/retrieved: {session_id}")
    else:
        print(f"   ✓ Django not available - Using session ID: {session_id}")
        print(f"   ⚠ Note: Session '{session_id}' must already exist in the database!")
        print(f"      Create it via Django admin or ensure it exists before posting data.")
    
    # Generate points
    print("\n2. Generating 100 data points along 100m path...")
    points = generate_path_points(num_points=100, path_length=100.0)
    print(f"   ✓ Generated {len(points)} points")
    
    # Display API URL being used
    print(f"\n3. API Configuration:")
    print(f"   Base URL: {base_url}")
    print(f"   Endpoints:")
    print(f"     - {base_url}/local-position-odom/")
    print(f"     - {base_url}/gps-fix-raw/")
    print(f"     - {base_url}/gps-fix-estimated/")
    
    # Test connectivity
    print(f"\n4. Testing API connectivity...", end='', flush=True)
    try:
        # Try to reach the server (just check if it's reachable)
        test_response = requests.get(
            base_url.replace('/api/telemetry', '/admin/'),
            timeout=3.0,
            allow_redirects=True
        )
        print(f" ✓ Connected (Status: {test_response.status_code})")
    except requests.exceptions.ConnectionError:
        print(f" ⚠ Connection failed - will attempt to post anyway")
        print(f"   Make sure the server is running at: {base_url}")
    except Exception as e:
        print(f" ⚠ Could not verify connectivity: {e}")
    
    # Post data
    print(f"\n5. Posting data via API...")
    results = post_telemetry_via_api(base_url, session_id, points)
    
    # Update session if Django is available
    if django_available and session:
        try:
            from django.utils import timezone
            session.end_time = timezone.now()
            session.duration_seconds = (session.end_time - session.start_time).total_seconds()
            session.total_telemetry_points = (
                results['local_odom']['success'] +
                results['gps_raw']['success'] +
                results['gps_estimated']['success']
            )
            session.save()
        except Exception as e:
            print(f"   ⚠ Could not update session: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Session ID: {session_id}")
    print(f"API Base URL: {base_url}")
    if django_available and session:
        print(f"Start Time: {session.start_time}")
        print(f"End Time: {session.end_time}")
        print(f"Duration: {session.duration_seconds:.2f} seconds")
    print(f"\nRecords Posted:")
    print(f"  • Local Position Odometry: {results['local_odom']['success']} (errors: {len(results['local_odom']['errors'])})")
    print(f"  • Raw GPS Fix: {results['gps_raw']['success']} (errors: {len(results['gps_raw']['errors'])})")
    print(f"  • Estimated GPS Fix: {results['gps_estimated']['success']} (errors: {len(results['gps_estimated']['errors'])})")
    total = (
        results['local_odom']['success'] +
        results['gps_raw']['success'] +
        results['gps_estimated']['success']
    )
    print(f"  • Total: {total}")
    
    if any(len(r['errors']) > 0 for r in results.values()):
        print(f"\nErrors encountered:")
        for key, value in results.items():
            if value['errors']:
                print(f"  {key}: {len(value['errors'])} errors")
                for err in value['errors'][:5]:  # Show first 5 errors
                    print(f"    - {err}")
    
    print("=" * 70)
    print("\n✓ Test data injection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

