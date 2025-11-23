"""
Serializers for telemetry API endpoints.
Handles validation and data transformation for telemetry data.
"""
from django.core.exceptions import ValidationError
from django.utils import timezone
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json


class TelemetrySerializer:
    """Base serializer for telemetry data"""
    
    @staticmethod
    def parse_timestamp(timestamp_value):
        """
        Parse timestamp from various formats:
        - ISO 8601 string
        - Unix timestamp (seconds or microseconds)
        - datetime object
        """
        if timestamp_value is None:
            return timezone.now()
        
        if isinstance(timestamp_value, datetime):
            if timezone.is_naive(timestamp_value):
                return timezone.make_aware(timestamp_value)
            return timestamp_value
        
        if isinstance(timestamp_value, (int, float)):
            # Assume microseconds if > 1e10, else seconds
            if timestamp_value > 1e10:
                # Microseconds since epoch
                return timezone.datetime.fromtimestamp(timestamp_value / 1e6, tz=timezone.utc)
            else:
                # Seconds since epoch
                return timezone.datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        
        if isinstance(timestamp_value, str):
            try:
                # Try ISO 8601 format
                return timezone.datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            except ValueError:
                # Try parsing as float/int
                try:
                    ts = float(timestamp_value)
                    if ts > 1e10:
                        return timezone.datetime.fromtimestamp(ts / 1e6, tz=timezone.utc)
                    else:
                        return timezone.datetime.fromtimestamp(ts, tz=timezone.utc)
                except (ValueError, OSError):
                    raise ValidationError(f"Invalid timestamp format: {timestamp_value}")
        
        raise ValidationError(f"Unsupported timestamp type: {type(timestamp_value)}")
    
    @staticmethod
    def parse_decimal(value, field_name="value"):
        """Parse decimal value, handling strings and floats"""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid {field_name}: {value}")


class LocalPositionOdomSerializer(TelemetrySerializer):
    """Serializer for LocalPositionOdom model"""
    
    REQUIRED_FIELDS = ['session_id', 'x', 'y', 'z', 'timestamp']
    OPTIONAL_FIELDS = [
        'timestamp_usec', 'vx', 'vy', 'vz', 'heading', 'heading_rate',
        'position_covariance', 'velocity_covariance',
        'xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid', 'heading_valid',
        'ref_timestamp', 'ref_lat', 'ref_lon', 'ref_alt',
        'dist_bottom', 'dist_bottom_valid', 'dist_bottom_sensor_bitfield',
        'eph', 'epv', 'evh', 'evv'
    ]
    
    @classmethod
    def validate_and_parse(cls, data):
        """Validate and parse LocalPositionOdom data"""
        errors = {}
        parsed_data = {}
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                errors[field] = f"This field is required."
            else:
                parsed_data[field] = data[field]
        
        if errors:
            return None, errors
        
        # Parse session_id
        from dreams_laboratory.models import DroneTelemetrySession
        try:
            session = DroneTelemetrySession.objects.get(session_id=parsed_data['session_id'])
            parsed_data['session'] = session
        except DroneTelemetrySession.DoesNotExist:
            errors['session_id'] = f"Session '{parsed_data['session_id']}' not found."
            return None, errors
        
        # Parse timestamp
        try:
            parsed_data['timestamp'] = cls.parse_timestamp(parsed_data['timestamp'])
        except ValidationError as e:
            errors['timestamp'] = str(e)
            return None, errors
        
        # Parse timestamp_usec if provided
        if 'timestamp_usec' in data:
            try:
                parsed_data['timestamp_usec'] = int(data['timestamp_usec'])
            except (ValueError, TypeError):
                errors['timestamp_usec'] = "Must be an integer."
        
        # Parse numeric fields
        numeric_fields = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'heading', 'heading_rate',
                         'ref_lat', 'ref_lon', 'ref_alt', 'dist_bottom',
                         'eph', 'epv', 'evh', 'evv']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    parsed_data[field] = float(data[field])
                except (ValueError, TypeError):
                    errors[field] = f"Must be a number."
        
        # Parse boolean fields
        boolean_fields = ['xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid', 
                         'heading_valid', 'dist_bottom_valid']
        for field in boolean_fields:
            if field in data:
                parsed_data[field] = bool(data[field])
        
        # Parse integer fields
        integer_fields = ['ref_timestamp', 'dist_bottom_sensor_bitfield']
        for field in integer_fields:
            if field in data and data[field] is not None:
                try:
                    parsed_data[field] = int(data[field])
                except (ValueError, TypeError):
                    errors[field] = f"Must be an integer."
        
        # Parse JSON fields (covariance matrices)
        json_fields = ['position_covariance', 'velocity_covariance']
        for field in json_fields:
            if field in data:
                if isinstance(data[field], str):
                    try:
                        parsed_data[field] = json.loads(data[field])
                    except json.JSONDecodeError:
                        errors[field] = f"Invalid JSON format."
                elif isinstance(data[field], list):
                    parsed_data[field] = data[field]
                elif data[field] is None:
                    parsed_data[field] = []
                else:
                    errors[field] = f"Must be a list or JSON string."
        
        if errors:
            return None, errors
        
        return parsed_data, None


class GPSFixRawSerializer(TelemetrySerializer):
    """Serializer for GPSFixRaw model"""
    
    REQUIRED_FIELDS = ['session_id', 'latitude', 'longitude', 'altitude', 'fix_type', 'timestamp']
    OPTIONAL_FIELDS = [
        'timestamp_usec', 'satellites_visible', 'satellites_used',
        'hdop', 'vdop', 'pdop', 'eph', 'epv', 's_variance_m_s',
        'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s', 'vel_m_s', 'cog_rad',
        'time_utc_usec', 'noise_per_ms', 'jamming_indicator', 'jamming_state',
        'device_id'
    ]
    
    @classmethod
    def validate_and_parse(cls, data):
        """Validate and parse GPSFixRaw data"""
        errors = {}
        parsed_data = {}
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                errors[field] = f"This field is required."
            else:
                parsed_data[field] = data[field]
        
        if errors:
            return None, errors
        
        # Parse session_id
        from dreams_laboratory.models import DroneTelemetrySession
        try:
            session = DroneTelemetrySession.objects.get(session_id=parsed_data['session_id'])
            parsed_data['session'] = session
        except DroneTelemetrySession.DoesNotExist:
            errors['session_id'] = f"Session '{parsed_data['session_id']}' not found."
            return None, errors
        
        # Parse timestamp
        try:
            parsed_data['timestamp'] = cls.parse_timestamp(parsed_data['timestamp'])
        except ValidationError as e:
            errors['timestamp'] = str(e)
            return None, errors
        
        # Parse latitude and longitude as Decimal
        try:
            parsed_data['latitude'] = cls.parse_decimal(parsed_data['latitude'], 'latitude')
            parsed_data['longitude'] = cls.parse_decimal(parsed_data['longitude'], 'longitude')
        except ValidationError as e:
            errors[str(e).split(':')[0].replace('Invalid ', '')] = str(e).split(':')[1].strip()
            return None, errors
        
        # Validate latitude/longitude ranges
        if parsed_data['latitude'] < Decimal('-90') or parsed_data['latitude'] > Decimal('90'):
            errors['latitude'] = "Must be between -90 and 90."
        if parsed_data['longitude'] < Decimal('-180') or parsed_data['longitude'] > Decimal('180'):
            errors['longitude'] = "Must be between -180 and 180."
        
        # Parse numeric fields
        numeric_fields = ['altitude', 'hdop', 'vdop', 'pdop', 'eph', 'epv', 's_variance_m_s',
                         'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s', 'vel_m_s', 'cog_rad']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    parsed_data[field] = float(data[field])
                except (ValueError, TypeError):
                    errors[field] = f"Must be a number."
        
        # Parse integer fields
        integer_fields = ['fix_type', 'satellites_visible', 'satellites_used',
                         'timestamp_usec', 'time_utc_usec', 'noise_per_ms',
                         'jamming_indicator', 'jamming_state', 'device_id']
        for field in integer_fields:
            if field in data and data[field] is not None:
                try:
                    parsed_data[field] = int(data[field])
                except (ValueError, TypeError):
                    errors[field] = f"Must be an integer."
        
        # Validate fix_type
        if 'fix_type' in parsed_data:
            if parsed_data['fix_type'] not in [0, 1, 2, 3, 4, 5]:
                errors['fix_type'] = "Must be 0-5 (0=no fix, 1=dead reckoning, 2=2D, 3=3D, 4=GPS+DR, 5=Time only)."
        
        if errors:
            return None, errors
        
        return parsed_data, None


class GPSFixEstimatedSerializer(TelemetrySerializer):
    """Serializer for GPSFixEstimated model"""
    
    REQUIRED_FIELDS = ['session_id', 'latitude', 'longitude', 'altitude', 'timestamp']
    OPTIONAL_FIELDS = [
        'timestamp_usec', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s',
        'position_covariance', 'velocity_covariance',
        'eph', 'epv', 'evh', 'evv', 'estimator_type', 'confidence',
        'position_valid', 'velocity_valid',
        'raw_gps_fix_id', 'local_position_id'
    ]
    
    @classmethod
    def validate_and_parse(cls, data):
        """Validate and parse GPSFixEstimated data"""
        errors = {}
        parsed_data = {}
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                errors[field] = f"This field is required."
            else:
                parsed_data[field] = data[field]
        
        if errors:
            return None, errors
        
        # Parse session_id
        from dreams_laboratory.models import DroneTelemetrySession
        try:
            session = DroneTelemetrySession.objects.get(session_id=parsed_data['session_id'])
            parsed_data['session'] = session
        except DroneTelemetrySession.DoesNotExist:
            errors['session_id'] = f"Session '{parsed_data['session_id']}' not found."
            return None, errors
        
        # Parse timestamp
        try:
            parsed_data['timestamp'] = cls.parse_timestamp(parsed_data['timestamp'])
        except ValidationError as e:
            errors['timestamp'] = str(e)
            return None, errors
        
        # Parse latitude and longitude as Decimal
        try:
            parsed_data['latitude'] = cls.parse_decimal(parsed_data['latitude'], 'latitude')
            parsed_data['longitude'] = cls.parse_decimal(parsed_data['longitude'], 'longitude')
        except ValidationError as e:
            field_name = str(e).split(':')[0].replace('Invalid ', '')
            errors[field_name] = str(e).split(':')[1].strip()
            return None, errors
        
        # Validate latitude/longitude ranges
        if parsed_data['latitude'] < Decimal('-90') or parsed_data['latitude'] > Decimal('90'):
            errors['latitude'] = "Must be between -90 and 90."
        if parsed_data['longitude'] < Decimal('-180') or parsed_data['longitude'] > Decimal('180'):
            errors['longitude'] = "Must be between -180 and 180."
        
        # Parse numeric fields
        numeric_fields = ['altitude', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s',
                         'eph', 'epv', 'evh', 'evv', 'confidence']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    parsed_data[field] = float(data[field])
                except (ValueError, TypeError):
                    errors[field] = f"Must be a number."
        
        # Parse boolean fields
        boolean_fields = ['position_valid', 'velocity_valid']
        for field in boolean_fields:
            if field in data:
                parsed_data[field] = bool(data[field])
        
        # Parse integer fields
        if 'timestamp_usec' in data and data['timestamp_usec'] is not None:
            try:
                parsed_data['timestamp_usec'] = int(data['timestamp_usec'])
            except (ValueError, TypeError):
                errors['timestamp_usec'] = "Must be an integer."
        
        # Parse string fields
        if 'estimator_type' in data:
            parsed_data['estimator_type'] = str(data['estimator_type'])
        
        # Parse JSON fields (covariance matrices)
        json_fields = ['position_covariance', 'velocity_covariance']
        for field in json_fields:
            if field in data:
                if isinstance(data[field], str):
                    try:
                        parsed_data[field] = json.loads(data[field])
                    except json.JSONDecodeError:
                        errors[field] = f"Invalid JSON format."
                elif isinstance(data[field], list):
                    parsed_data[field] = data[field]
                elif data[field] is None:
                    parsed_data[field] = []
                else:
                    errors[field] = f"Must be a list or JSON string."
        
        # Parse related object IDs
        if 'raw_gps_fix_id' in data and data['raw_gps_fix_id']:
            from dreams_laboratory.models import GPSFixRaw
            try:
                parsed_data['raw_gps_fix'] = GPSFixRaw.objects.get(id=int(data['raw_gps_fix_id']))
            except (GPSFixRaw.DoesNotExist, ValueError, TypeError):
                errors['raw_gps_fix_id'] = "Invalid raw GPS fix ID."
        
        if 'local_position_id' in data and data['local_position_id']:
            from dreams_laboratory.models import LocalPositionOdom
            try:
                parsed_data['local_position'] = LocalPositionOdom.objects.get(id=int(data['local_position_id']))
            except (LocalPositionOdom.DoesNotExist, ValueError, TypeError):
                errors['local_position_id'] = "Invalid local position ID."
        
        if errors:
            return None, errors
        
        return parsed_data, None

