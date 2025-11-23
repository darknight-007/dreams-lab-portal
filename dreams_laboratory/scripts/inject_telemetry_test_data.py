#!/usr/bin/env python3
"""
Script to inject 100 test telemetry data points along a 100m path in Tempe Town Lake, Arizona.
Data collected by robotic boat "RV Karin Valentine".

Usage:
    python manage.py shell < scripts/inject_telemetry_test_data.py
    OR
    python scripts/inject_telemetry_test_data.py (if run from project root with Django setup)
"""

import os
import sys
import django
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
import math
import random

# Setup Django environment
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dreams_laboratory.settings')
django.setup()

from django.utils import timezone
from dreams_laboratory.models import (
    Asset, People, Project, DroneTelemetrySession,
    LocalPositionOdom, GPSFixRaw, GPSFixEstimated
)


def get_or_create_test_objects():
    """Get or create necessary Person, Project, and Asset objects"""
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
    
    # Update asset if it exists but doesn't have person/project
    if not asset.person or not asset.project:
        asset.person = person
        asset.project = project
        asset.save()
    
    return person, project, asset


def generate_path_points(num_points=100, path_length=100.0):
    """
    Generate points along a 100m path in Tempe Town Lake.
    Creates a gentle curved path moving east-west along the lake.
    
    Tempe Town Lake coordinates: ~33.4255° N, 111.9400° W
    """
    # Starting point (west end of path)
    start_lat = Decimal('33.4255000')
    start_lon = Decimal('-111.9405000')  # Slightly west
    start_alt = 350.0  # Meters above sea level
    
    # Calculate meters per degree (approximate for Tempe)
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    lat_meters_per_degree = 111000.0
    lon_meters_per_degree = 111000.0 * math.cos(math.radians(float(start_lat)))
    
    points = []
    base_time = timezone.now()
    
    for i in range(num_points):
        # Progress along path (0 to 1)
        progress = i / (num_points - 1) if num_points > 1 else 0
        
        # Create a gentle S-curve path (east-west movement)
        # Add some north-south variation for realism
        distance_east = path_length * progress  # Total distance east
        distance_north = 5.0 * math.sin(progress * math.pi * 2)  # Slight north-south curve
        
        # Convert to lat/lon
        lat_offset = distance_north / lat_meters_per_degree
        lon_offset = distance_east / lon_meters_per_degree
        
        lat = start_lat + Decimal(str(lat_offset))
        lon = start_lon + Decimal(str(lon_offset))
        
        # Add small random noise for realism (±0.5m)
        lat_noise = Decimal(str(random.uniform(-0.5, 0.5) / lat_meters_per_degree))
        lon_noise = Decimal(str(random.uniform(-0.5, 0.5) / lon_meters_per_degree))
        
        lat += lat_noise
        lon += lon_noise
        
        # Calculate local NED frame (relative to start point)
        # NED: North-East-Down (positive down)
        x = distance_north + random.uniform(-0.1, 0.1)  # North (m)
        y = distance_east + random.uniform(-0.1, 0.1)   # East (m)
        z = -0.5 + random.uniform(-0.2, 0.2)  # Down (m, boat slightly below water surface)
        
        # Calculate velocity (boat moving at ~1 m/s)
        speed = 1.0 + random.uniform(-0.2, 0.2)  # m/s
        heading = math.atan2(distance_east, distance_north) if i > 0 else 0.0
        
        vx = speed * math.cos(heading) + random.uniform(-0.1, 0.1)
        vy = speed * math.sin(heading) + random.uniform(-0.1, 0.1)
        vz = random.uniform(-0.05, 0.05)  # Minimal vertical velocity
        
        # Timestamp (1 second intervals)
        timestamp = base_time + timedelta(seconds=i)
        timestamp_usec = int(timestamp.timestamp() * 1e6)
        
        points.append({
            'index': i,
            'timestamp': timestamp,
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
        })
    
    return points


def create_telemetry_data(session, points):
    """Create telemetry records for all points"""
    created_count = {
        'local_odom': 0,
        'gps_raw': 0,
        'gps_estimated': 0
    }
    
    print(f"\nCreating telemetry data for {len(points)} points...")
    
    for i, point in enumerate(points):
        if (i + 1) % 10 == 0:
            print(f"  Processing point {i + 1}/{len(points)}...")
        
        # Reference frame for local odom (start point)
        ref_lat = float(points[0]['lat'])
        ref_lon = float(points[0]['lon'])
        ref_alt = points[0]['alt']
        
        # 1. Create LocalPositionOdom
        try:
            local_odom = LocalPositionOdom.objects.create(
                session=session,
                timestamp=point['timestamp'],
                timestamp_usec=point['timestamp_usec'],
                x=point['x'],
                y=point['y'],
                z=point['z'],
                vx=point['vx'],
                vy=point['vy'],
                vz=point['vz'],
                heading=point['heading'],
                heading_rate=random.uniform(-0.05, 0.05),
                xy_valid=True,
                z_valid=True,
                v_xy_valid=True,
                v_z_valid=True,
                heading_valid=True,
                position_covariance=[
                    0.01, 0, 0,
                    0, 0.01, 0,
                    0, 0, 0.02
                ],
                velocity_covariance=[
                    0.05, 0, 0,
                    0, 0.05, 0,
                    0, 0, 0.1
                ],
                ref_lat=ref_lat,
                ref_lon=ref_lon,
                ref_alt=ref_alt,
                eph=random.uniform(0.3, 0.8),
                epv=random.uniform(0.5, 1.2),
                evh=random.uniform(0.1, 0.3),
                evv=random.uniform(0.15, 0.4)
            )
            created_count['local_odom'] += 1
        except Exception as e:
            print(f"  Error creating LocalPositionOdom for point {i}: {e}")
            continue
        
        # 2. Create GPSFixRaw (with some noise)
        try:
            # Add GPS noise (raw GPS is less accurate)
            gps_noise_lat = random.uniform(-2.0, 2.0) / 111000.0  # ~2m noise
            gps_noise_lon = random.uniform(-2.0, 2.0) / (111000.0 * math.cos(math.radians(float(point['lat']))))
            
            gps_raw = GPSFixRaw.objects.create(
                session=session,
                timestamp=point['timestamp'],
                timestamp_usec=point['timestamp_usec'],
                latitude=point['lat'] + Decimal(str(gps_noise_lat)),
                longitude=point['lon'] + Decimal(str(gps_noise_lon)),
                altitude=point['alt'] + random.uniform(-1.0, 1.0),
                fix_type=3,  # 3D fix
                satellites_visible=random.randint(8, 14),
                satellites_used=random.randint(7, 12),
                hdop=random.uniform(0.8, 2.0),
                vdop=random.uniform(1.0, 2.5),
                pdop=random.uniform(1.5, 3.0),
                eph=random.uniform(1.5, 3.5),  # Raw GPS less accurate
                epv=random.uniform(2.0, 4.0),
                s_variance_m_s=random.uniform(0.3, 0.8),
                vel_n_m_s=point['vx'] + random.uniform(-0.2, 0.2),
                vel_e_m_s=point['vy'] + random.uniform(-0.2, 0.2),
                vel_d_m_s=point['vz'] + random.uniform(-0.1, 0.1),
                vel_m_s=point['speed'] + random.uniform(-0.2, 0.2),
                cog_rad=point['heading'] + random.uniform(-0.1, 0.1),
                time_utc_usec=point['timestamp_usec'],
                noise_per_ms=random.randint(20, 40),
                jamming_indicator=random.randint(0, 10),
                jamming_state=1,  # OK
                device_id=1
            )
            created_count['gps_raw'] += 1
        except Exception as e:
            print(f"  Error creating GPSFixRaw for point {i}: {e}")
            continue
        
        # 3. Create GPSFixEstimated (filtered, more accurate)
        try:
            # Estimated GPS is more accurate (filtered)
            est_noise_lat = random.uniform(-0.5, 0.5) / 111000.0  # ~0.5m noise
            est_noise_lon = random.uniform(-0.5, 0.5) / (111000.0 * math.cos(math.radians(float(point['lat']))))
            
            gps_est = GPSFixEstimated.objects.create(
                session=session,
                timestamp=point['timestamp'],
                timestamp_usec=point['timestamp_usec'],
                latitude=point['lat'] + Decimal(str(est_noise_lat)),
                longitude=point['lon'] + Decimal(str(est_noise_lon)),
                altitude=point['alt'] + random.uniform(-0.3, 0.3),
                vel_n_m_s=point['vx'] + random.uniform(-0.1, 0.1),
                vel_e_m_s=point['vy'] + random.uniform(-0.1, 0.1),
                vel_d_m_s=point['vz'] + random.uniform(-0.05, 0.05),
                position_covariance=[
                    0.3, 0, 0,
                    0, 0.3, 0,
                    0, 0, 0.8
                ],
                velocity_covariance=[
                    0.08, 0, 0,
                    0, 0.08, 0,
                    0, 0, 0.15
                ],
                eph=random.uniform(0.8, 1.8),  # Better than raw GPS
                epv=random.uniform(1.2, 2.2),
                evh=random.uniform(0.08, 0.2),
                evv=random.uniform(0.12, 0.3),
                estimator_type='EKF2',
                confidence=random.uniform(0.85, 0.98),
                position_valid=True,
                velocity_valid=True,
                raw_gps_fix=gps_raw,
                local_position=local_odom
            )
            created_count['gps_estimated'] += 1
        except Exception as e:
            print(f"  Error creating GPSFixEstimated for point {i}: {e}")
            continue
    
    return created_count


def main():
    """Main function to inject test telemetry data"""
    print("=" * 70)
    print("Tempe Town Lake Telemetry Data Injection")
    print("Robotic Boat: RV Karin Valentine")
    print("=" * 70)
    
    # Get or create necessary objects
    print("\n1. Setting up test objects...")
    person, project, asset = get_or_create_test_objects()
    print(f"   ✓ Person: {person.first_name} {person.last_name}")
    print(f"   ✓ Project: {project.title}")
    print(f"   ✓ Asset: {asset.asset_name}")
    
    # Create telemetry session
    print("\n2. Creating telemetry session...")
    session_id = f"tempe_town_lake_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = timezone.now()
    
    session, created = DroneTelemetrySession.objects.get_or_create(
        session_id=session_id,
        defaults={
            'asset': asset,
            'project': project,
            'start_time': start_time,
            'flight_mode': 'AUTO',
            'mission_type': 'Lake Survey',
            'notes': 'Test data injection: 100 points along 100m path in Tempe Town Lake, Arizona'
        }
    )
    
    if created:
        print(f"   ✓ Created session: {session_id}")
    else:
        print(f"   ✓ Using existing session: {session_id}")
    
    # Generate path points
    print("\n3. Generating 100 data points along 100m path...")
    points = generate_path_points(num_points=100, path_length=100.0)
    print(f"   ✓ Generated {len(points)} points")
    print(f"   ✓ Path: {points[0]['lat']:.6f}°N, {points[0]['lon']:.6f}°W to {points[-1]['lat']:.6f}°N, {points[-1]['lon']:.6f}°W")
    
    # Create telemetry data
    print("\n4. Creating telemetry records...")
    created_count = create_telemetry_data(session, points)
    
    # Update session end time and duration
    end_time = timezone.now()
    duration = (end_time - start_time).total_seconds()
    session.end_time = end_time
    session.duration_seconds = duration
    session.total_telemetry_points = (
        created_count['local_odom'] + 
        created_count['gps_raw'] + 
        created_count['gps_estimated']
    )
    session.save()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Session ID: {session.session_id}")
    print(f"Asset: {session.asset.asset_name}")
    print(f"Start Time: {session.start_time}")
    print(f"End Time: {session.end_time}")
    print(f"Duration: {session.duration_seconds:.2f} seconds")
    print(f"\nRecords Created:")
    print(f"  • Local Position Odometry: {created_count['local_odom']}")
    print(f"  • Raw GPS Fix: {created_count['gps_raw']}")
    print(f"  • Estimated GPS Fix: {created_count['gps_estimated']}")
    print(f"  • Total: {session.total_telemetry_points}")
    print(f"\nPath Details:")
    print(f"  • Start: {points[0]['lat']:.7f}°N, {points[0]['lon']:.7f}°W")
    print(f"  • End: {points[-1]['lat']:.7f}°N, {points[-1]['lon']:.7f}°W")
    print(f"  • Distance: ~100 meters")
    print(f"  • Location: Tempe Town Lake, Arizona")
    print("=" * 70)
    print("\n✓ Test data injection complete!")
    print(f"\nView the data in Django admin:")
    print(f"  http://deepgis.org/admin/dreams_laboratory/dronetelemetrysession/")
    print("=" * 70)


if __name__ == '__main__':
    main()

