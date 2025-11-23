#!/usr/bin/env python3
"""
Script to inject 100 test GPS raw telemetry data points via REST API.
Posts only GPS raw data to the API endpoint.

This script is standalone and does NOT require Django - only needs the 'requests' library.

Usage:
    python scripts/inject_telemetry_via_api.py [--url API_URL] [--session-id SESSION_ID]

Configuration:
    Set API_BASE_URL below or pass --url argument
    Session ID will be randomly generated if not provided (session must exist in DB)
"""

import sys
from datetime import datetime, timedelta
import math
import random
import requests
import json
import argparse
import string

# ============================================================================
# CONFIGURATION - Set your API base URL here
# ============================================================================
# Default API base URL (can be overridden via command-line argument)
API_BASE_URL = "https://deepgis.org/api/telemetry"

# Alternative URLs (uncomment to use):
# API_BASE_URL = "http://localhost:8000/api/telemetry"  # Local development
# API_BASE_URL = "https://deepgis.org/api/telemetry"   # Production HTTPS
# API_BASE_URL = "http://172.20.0.10/api/telemetry"    # Docker internal network


def generate_random_session_id():
    """Generate a random session ID"""
    import string
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f"test_session_{timestamp}_{random_suffix}"


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


def post_gps_raw_via_api(base_url, session_id, points):
    """Post GPS raw telemetry data via REST API"""
    results = {
        'gps_raw': {'success': 0, 'errors': []}
    }
    
    print(f"\nPosting GPS raw telemetry data via API for {len(points)} points...")
    
    for i, point in enumerate(points):
        if (i + 1) % 10 == 0:
            print(f"  Processing point {i + 1}/{len(points)}...")
        
        # Post GPSFixRaw
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
                results['gps_raw']['success'] += 1
            else:
                error_data = response.json() if response.content else {'message': 'Unknown error'}
                results['gps_raw']['errors'].append({
                    'index': i,
                    'status': response.status_code,
                    'error': error_data
                })
        except Exception as e:
            results['gps_raw']['errors'].append({
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
        help='Session ID to use (must exist in database). If not provided, a random session ID will be generated.'
    )
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')  # Remove trailing slash if present
    
    print("=" * 70)
    print("Tempe Town Lake Telemetry Data Injection (via API)")
    print("Robotic Boat: RV Karin Valentine")
    print("=" * 70)
    
    # Generate or use provided session ID
    print("\n1. Setting up session...")
    if args.session_id:
        session_id = args.session_id
        print(f"   ✓ Using provided session ID: {session_id}")
    else:
        session_id = generate_random_session_id()
        print(f"   ✓ Generated random session ID: {session_id}")
    
    print(f"   ⚠ Note: Session '{session_id}' must exist in the database!")
    print(f"      Create it via Django admin before posting data, or the API will return validation errors.")
    
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
    print(f"\n5. Posting GPS raw data via API...")
    results = post_gps_raw_via_api(base_url, session_id, points)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Session ID: {session_id}")
    print(f"API Base URL: {base_url}")
    print(f"\nRecords Posted:")
    print(f"  • Raw GPS Fix: {results['gps_raw']['success']} (errors: {len(results['gps_raw']['errors'])})")
    print(f"  • Total: {results['gps_raw']['success']}")
    
    if len(results['gps_raw']['errors']) > 0:
        print(f"\nErrors encountered:")
        print(f"  GPS Raw: {len(results['gps_raw']['errors'])} errors")
        for err in results['gps_raw']['errors'][:5]:  # Show first 5 errors
            print(f"    - {err}")
    
    print("=" * 70)
    print("\n✓ Test data injection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

