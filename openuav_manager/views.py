from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import subprocess
import json
import time
from datetime import datetime
from .models import Container
import logging
import uuid
import psutil
from django.conf import settings
import os
import random
import string

logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and return output"""
    logger.info(f"Executing command: {command}")
    try:
        # Always use shell=True for Docker commands to handle format strings properly
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        logger.info(f"Command output - stdout: {result.stdout.strip()}")
        if result.stderr:
            logger.error(f"Command error - stderr: {result.stderr.strip()}")
        logger.info(f"Command return code: {result.returncode}")
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        logger.error(f"Exception running command: {str(e)}")
        return "", str(e), 1

def sync_containers():
    """Sync container status with database"""
    try:
        # Get list of all containers with openuav prefix
        stdout, stderr, code = run_command("docker ps -a --filter name=openuav --format '{{.ID}}'")
        if code != 0:
            logger.error(f"Error getting container list: {stderr}")
            return
        
        # Get container IDs
        container_ids = [id.strip() for id in stdout.split('\n') if id.strip()]
        
        # Get detailed container info using inspect
        containers = {}
        for container_id in container_ids:
            inspect_cmd = f"""docker inspect --format='{{{{.Id}}}}\\t{{{{.Name}}}}\\t{{{{.State.Status}}}}\\t{{{{.Created}}}}\\t{{{{.Config.Image}}}}' {container_id}"""
            stdout, stderr, code = run_command(inspect_cmd)
            if code == 0:
                full_id, name, status, created, image = stdout.strip().split('\t')
                # Remove leading slash from name
                name = name.lstrip('/')
                containers[full_id] = {
                    'full_id': full_id,
                    'short_id': container_id,
                    'name': name,
                    'status': status,
                    'created': created,
                    'image': image
                }
        
        # Get all container records from database
        db_containers = Container.objects.all()
        db_container_ids = set(db_containers.values_list('container_id', flat=True))
        current_container_ids = set(containers.keys())
        
        # Remove stale records (containers that no longer exist)
        stale_ids = db_container_ids - current_container_ids
        if stale_ids:
            Container.objects.filter(container_id__in=stale_ids).delete()
            logger.info(f"Removed {len(stale_ids)} stale container records")
        
        # Update or create records for existing containers
        for container_id, container_data in containers.items():
            # Update or create container record
            Container.objects.update_or_create(
                container_id=container_data['full_id'],
                defaults={
                    'name': container_data['name'],
                    'status': container_data['status'],
                    'image': container_data['image'],
                    'short_id': container_data['short_id']
                }
            )
        
        logger.info(f"Successfully synced {len(containers)} containers")
    except Exception as e:
        logger.error(f"Error in sync_containers: {str(e)}", exc_info=True)

def container_list(request):
    """Display consolidated OpenUAV management interface"""
    try:
        sync_containers()
        containers = Container.objects.all().order_by('-created')
        
        # Get filtered container counts
        running_count = containers.filter(status='running').count()
        stopped_count = containers.filter(status='stopped').count()
        
        # Get system stats for initial display
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_stats = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': round(memory.used / (1024**3), 2),
            'memory_total': round(memory.total / (1024**3), 2),
            'disk_percent': disk.percent,
            'disk_used': round(disk.used / (1024**3), 2),
            'disk_total': round(disk.total / (1024**3), 2)
        }
        
        return render(request, 'openuav_manager/container_list.html', {
            'containers': containers,
            'system_stats': system_stats,
            'running_count': running_count,
            'stopped_count': stopped_count
        })
    except Exception as e:
        logger.error(f"Error in container_list: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return render(request, 'openuav_manager/container_list.html', {
            'containers': [],
            'system_stats': {},
            'running_count': 0,
            'stopped_count': 0
        })

def container_action(request, container_id, action):
    """Handle container actions (start/stop/delete)"""
    try:
        if action == 'start':
            stdout, stderr, code = run_command(f"docker start {container_id}")
            if code == 0:
                messages.success(request, f'Container {container_id} started successfully')
            else:
                messages.error(request, f'Error starting container: {stderr}')
        elif action == 'stop':
            # First verify container exists and is running
            container = Container.objects.filter(container_id=container_id, status='running').first()
            if not container:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Container not found or not running'
                }, status=404)

            logger.info(f"Stopping container: {container.name}")
            stdout, stderr, code = run_command(f"docker stop {container_id}")
            if code == 0:
                # Update container status in database
                container.status = 'stopped'
                container.save()
                messages.success(request, f'Container {container_id} stopped successfully')
                
                # Clean up X11 socket and VNC files
                try:
                    run_command("rm -f /tmp/.X11-unix/X1")
                    run_command("rm -f /tmp/.X1*")
                    run_command("pkill -f Xvnc || true")
                    run_command("pkill -f vncserver || true")
                except Exception as e:
                    logger.warning(f"Error during cleanup: {str(e)}")
            else:
                messages.error(request, f'Error stopping container: {stderr}')
        elif action == 'delete':
            # First stop the container if it's running
            container = Container.objects.filter(container_id=container_id).first()
            if container and container.status == 'running':
                run_command(f"docker stop {container_id}")
            # Then remove it
            stdout, stderr, code = run_command(f"docker rm {container_id}")
            if code == 0:
                messages.success(request, f'Container {container_id} deleted successfully')
                # Remove from database
                if container:
                    container.delete()
            else:
                messages.error(request, f'Error deleting container: {stderr}')
        
        # Allow time for container status to update
        import time
        time.sleep(1)
        sync_containers()
        
    except Exception as e:
        logger.error(f"Error in container_action: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success'})
    return redirect('container_list')

@csrf_exempt
@require_http_methods(["POST"])
def launch_openuav(request):
    """Launch a new OpenUAV container instance."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    try:
        # Clean up old containers first
        stdout, stderr, code = run_command("docker ps -a --filter name=digital-twin --format '{{.ID}}' | xargs -r docker rm -f")
        if code != 0:
            logger.warning(f"Error cleaning up old containers: {stderr}")

        # Generate unique container ID (12 chars)
        container_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        container_name = f'digital-twin-{container_id}'
        
        # Get absolute path to run_container.sh in openuav2 folder
        script_path = '/app/openuav2/run_container.sh'
        
        if not os.path.exists(script_path):
            logger.error(f"Container launch script not found at {script_path}")
            return JsonResponse({
                'error': 'Container launch script not found',
                'details': 'Internal server error'
            }, status=500)
        
        # Set environment variables for the script
        env = os.environ.copy()
        env['CONTAINER_NAME'] = container_name
        
        # Run the container launch script
        result = subprocess.run(
            ['bash', script_path],
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        
        try:
            # Parse the JSON output from the script
            container_info = json.loads(result.stdout)
            
            # Wait for container to be running with retries
            max_retries = 5
            retry_delay = 1  # seconds
            container_running = False
            
            for attempt in range(max_retries):
                check_result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True
                )
                
                if container_name in check_result.stdout:
                    container_running = True
                    break
                    
                logger.info(f"Container not running yet, attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
            
            if not container_running:
                logger.error(f"Container {container_name} not running after {max_retries} attempts")
                return JsonResponse({
                    'error': 'Container failed to start',
                    'details': 'Container is not running after launch'
                }, status=500)
            
            # Get full container info using inspect
            inspect_cmd = f"""docker inspect --format='{{{{.Id}}}}\\t{{{{.Name}}}}\\t{{{{.State.Status}}}}\\t{{{{.Created}}}}\\t{{{{.Config.Image}}}}' {container_info['container_id']}"""
            stdout, stderr, code = run_command(inspect_cmd)
            if code == 0:
                full_id, name, status, created, image = stdout.strip().split('\t')
                # Remove leading slash from name
                name = name.lstrip('/')
                
                # Create container record
                Container.objects.create(
                    container_id=full_id,
                    short_id=container_info['short_id'],
                    name=name,
                    status=status,
                    image=image,
                    ports={'vnc': container_info.get('vnc_port', 5901)},
                    vnc_url=container_info['vnc_url']
                )
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'OpenUAV instance launched successfully',
                    'container': {
                        'id': container_info['container_id'],
                        'short_id': container_info['short_id'],
                        'name': container_name,
                        'url': container_info['url'],
                        'vnc_url': container_info['vnc_url']
                    }
                })
            else:
                raise Exception(f"Failed to inspect container: {stderr}")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse container launch output: {result.stdout}")
            return JsonResponse({
                'error': 'Failed to parse container information',
                'details': 'Internal server error'
            }, status=500)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch container: {e.stderr}")
        return JsonResponse({
            'error': 'Failed to launch container',
            'details': e.stderr
        }, status=500)
    except Exception as e:
        logger.error(f"Unexpected error launching container: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': 'Internal server error',
            'details': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def stop_openuav(request):
    try:
        # Get container name from request data
        data = json.loads(request.body) if request.body else {}
        container_name = data.get('container_name')
        
        if not container_name:
            # Try to find the most recent container for this session
            container = Container.objects.filter(
                session_type='guest',  # For now, assume guest session
                status='running'
            ).order_by('-created').first()
            
            if container:
                container_name = container.name
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No running container found'
                }, status=404)
        
        logger.info(f"Stopping container: {container_name}")
        
        # First stop the container
        stdout, stderr, code = run_command(f"docker stop {container_name}")
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error stopping container: {stderr}'
            }, status=500)
            
        # Then remove the container
        stdout, stderr, code = run_command(f"docker rm {container_name}")
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error removing container: {stderr}'
            }, status=500)
            
        # Update container status in database
        try:
            container = Container.objects.get(name=container_name)
            container.status = 'removed'
            container.save()
        except Container.DoesNotExist:
            logger.warning(f"Container {container_name} not found in database")
            
        # Clean up X11 socket
        run_command("rm -f /tmp/.X11-unix/X1")
            
        return JsonResponse({
            'status': 'success',
            'message': 'OpenUAV instance stopped and removed successfully'
        })
    except Exception as e:
        logger.error(f"Error in stop_openuav: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def openuav_status(request):
    try:
        # Get container info
        stdout, stderr, code = run_command("docker ps --filter name=openuav --format '{{json .}}'")
        
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error getting container status: {stderr}'
            }, status=500)
        
        containers = [json.loads(line) for line in stdout.split('\n') if line]
        
        if not containers:
            return JsonResponse({
                'status': 'success',
                'is_running': False,
                'instance_count': 0
            })
        
        return JsonResponse({
            'status': 'success',
            'is_running': True,
            'instance_count': 1
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def container_logs(request, container_id):
    """Get container logs"""
    try:
        stdout, stderr, code = run_command(f"docker logs {container_id}")
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error getting logs: {stderr}'
            }, status=500)
        
        return JsonResponse({
            'status': 'success',
            'logs': stdout
        })
    except Exception as e:
        logger.error(f"Error getting container logs: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def system_stats(request):
    """Get system resource statistics"""
    try:
        # Get system stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get container stats
        container_stats = {}
        stdout, stderr, code = run_command("docker stats --no-stream --format '{{.ID}}\t{{.CPUPerc}}\t{{.MemPerc}}'")
        if code == 0:
            for line in stdout.splitlines():
                if line.strip():
                    container_id, cpu_perc, mem_perc = line.split('\t')
                    container_stats[container_id] = {
                        'cpu_percent': float(cpu_perc.strip('%')),
                        'memory_percent': float(mem_perc.strip('%'))
                    }
        
        return JsonResponse({
            'status': 'success',
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': round(memory.used / (1024**3), 2),  # GB
                'memory_total': round(memory.total / (1024**3), 2),  # GB
                'disk_percent': disk.percent,
                'disk_used': round(disk.used / (1024**3), 2),  # GB
                'disk_total': round(disk.total / (1024**3), 2)  # GB
            },
            'containers': container_stats
        })
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["POST"])
def save_config(request):
    """Save instance configuration"""
    try:
        config = json.loads(request.body)
        
        # Validate configuration
        required_fields = ['default_image', 'cpu_limit', 'memory_limit', 'vnc_resolution']
        if not all(field in config for field in required_fields):
            return JsonResponse({
                'status': 'error',
                'message': 'Missing required configuration fields'
            }, status=400)
        
        # Save configuration to settings or database
        # For now, we'll just validate and return success
        return JsonResponse({'status': 'success'})
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["POST"])
def batch_action(request):
    """Perform batch actions on containers"""
    try:
        data = json.loads(request.body)
        action = data.get('action')
        container_ids = data.get('containers', [])
        
        if not action:
            return JsonResponse({
                'status': 'error',
                'message': 'Missing action'
            }, status=400)
        
        if action == 'cleanup':
            # Special case: cleanup all digital-twin containers
            stdout, stderr, code = run_command("docker ps -a --filter name=digital-twin --format '{{.ID}}' | xargs -r docker rm -f")
            success = code == 0
            return JsonResponse({
                'status': 'success' if success else 'error',
                'message': stderr if not success else 'Cleanup successful'
            })
        
        if not container_ids:
            return JsonResponse({
                'status': 'error',
                'message': 'Missing container IDs'
            }, status=400)
        
        results = {}
        for container_id in container_ids:
            if action == 'start':
                stdout, stderr, code = run_command(f"docker start {container_id}")
            elif action == 'stop':
                stdout, stderr, code = run_command(f"docker stop {container_id}")
            elif action == 'delete':
                stdout, stderr, code = run_command(f"docker rm -f {container_id}")
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Invalid action: {action}'
                }, status=400)
            
            results[container_id] = {
                'success': code == 0,
                'message': stderr if code != 0 else 'Success'
            }
        
        return JsonResponse({
            'status': 'success',
            'results': results
        })
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Error performing batch action: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def container_status_update(request):
    """Get detailed status update of all OpenUAV containers and sync with database"""
    try:
        # Get list of all containers with openuav prefix
        stdout, stderr, code = run_command("docker ps -a --filter name=openuav --format '{{json .}}'")
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error getting container list: {stderr}'
            }, status=500)
        
        # Parse container data and build status maps
        current_containers = {}
        status_counts = {
            'running': 0,
            'stopped': 0,
            'created': 0,
            'restarting': 0,
            'removing': 0,
            'paused': 0,
            'dead': 0,
            'unknown': 0
        }
        
        for line in stdout.split('\n'):
            if line:
                try:
                    container_data = json.loads(line)
                    container_id = container_data.get('ID')
                    if container_id:
                        status = container_data.get('Status', '').lower()
                        
                        # Determine detailed status
                        if 'up' in status:
                            container_status = 'running'
                        elif 'exited' in status:
                            container_status = 'stopped'
                        elif 'created' in status:
                            container_status = 'created'
                        elif 'restarting' in status:
                            container_status = 'restarting'
                        elif 'removing' in status:
                            container_status = 'removing'
                        elif 'paused' in status:
                            container_status = 'paused'
                        elif 'dead' in status:
                            container_status = 'dead'
                        else:
                            container_status = 'unknown'
                        
                        current_containers[container_id] = {
                            'id': container_id,
                            'name': container_data.get('Names'),
                            'status': container_status,
                            'image': container_data.get('Image', ''),
                            'created': container_data.get('CreatedAt', ''),
                            'ports': container_data.get('Ports', ''),
                            'state': container_data.get('State', ''),
                            'status_age': container_data.get('Status', '').replace('Up ', '').replace('Exited ', '')
                        }
                        status_counts[container_status] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        # Get all container records from database
        db_containers = Container.objects.all()
        db_container_ids = set(db_containers.values_list('container_id', flat=True))
        current_container_ids = set(current_containers.keys())
        
        # Find containers to remove from database
        removed_ids = db_container_ids - current_container_ids
        removed_count = len(removed_ids)
        if removed_ids:
            Container.objects.filter(container_id__in=removed_ids).delete()
        
        # Update or create records for existing containers
        updated_count = 0
        created_count = 0
        for container_id, container_data in current_containers.items():
            container, created = Container.objects.update_or_create(
                container_id=container_id,
                defaults={
                    'name': container_data['name'],
                    'status': container_data['status'],
                    'image': container_data['image']
                }
            )
            if created:
                created_count += 1
            else:
                updated_count += 1
        
        return JsonResponse({
            'status': 'success',
            'containers': list(current_containers.values()),
            'status_counts': status_counts,
            'database_changes': {
                'removed': removed_count,
                'updated': updated_count,
                'created': created_count
            },
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in container_status_update: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
