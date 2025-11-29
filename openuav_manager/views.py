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
from .models import Container, RosbagFile, RosbagSession
import logging
import uuid
import psutil
from django.conf import settings
import os
import random
import string
import requests
from pathlib import Path
import yaml

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
    try:
        # Try to get data from POST first, then try JSON
        if request.POST:
            username = request.POST.get('username', '').strip()
            passcode = request.POST.get('passcode', '').strip()
        else:
            try:
                data = json.loads(request.body)
                username = data.get('username', '').strip()
                passcode = data.get('passcode', '').strip()
            except json.JSONDecodeError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid request format'
                }, status=400)

        # Validate inputs
        if not username or not passcode:
            return JsonResponse({
                'status': 'error',
                'message': 'Username and passcode are required'
            }, status=400)

        # Validate passcode
        if passcode != 'liftoff':
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid passcode'
            }, status=401)

        # Clean up old containers first
        stdout, stderr, code = run_command("docker ps -a --filter name=digital-twin --format '{{.ID}}' | xargs -r docker rm -f")
        if code != 0:
            logger.warning(f"Error cleaning up old containers: {stderr}")

        # Generate container name
        container_name = f'digital-twin-{username}'
        
        # Launch container with the specified format
        launch_cmd = f"""docker run --init --runtime=nvidia --network=dreamslab --privileged --name={container_name} --hostname={container_name} -d -e VGL_DISPLAY=:0.0 -e DISPLAY=:1.0 -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw digital-twin-ras-ses-598:5.0"""
        
        
        # Print the full command to console
        logger.info("Launching container with command:")
        logger.info("=" * 80)
        logger.info(launch_cmd)
        logger.info("=" * 80)
        
        stdout, stderr, code = run_command(launch_cmd)
        if code != 0:
            logger.error(f"Failed to launch container: {stderr}")
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to launch container: {stderr}'
            }, status=500)
            
        # Get container ID from stdout
        container_id = stdout.strip()
        
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
        
        # Simple container inspection
        inspect_cmd = f"docker inspect --format='{{{{.Name}}}} {{{{.State.Status}}}}' {container_name}"
        stdout, stderr, code = run_command(inspect_cmd)
        if code == 0:
            # Create container record
            Container.objects.create(
                container_id=container_id,
                short_id=container_id[:12],
                name=container_name,
                status='running',
                image='digital-twin-ras-ses-598:5.0',
                ports={'vnc': 5901},  # Default VNC port
                vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
            )
            
            return JsonResponse({
                'status': 'success',
                'message': 'OpenUAV instance launched successfully',
                'container': {
                    'id': container_id,
                    'short_id': container_id[:12],
                    'name': container_name,
                    'vnc_url': f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
                }
            })
        else:
            raise Exception(f"Failed to inspect container: {stderr}")
            
    except Exception as e:
        logger.error(f"Unexpected error launching container: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': 'Internal server error',
            'details': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def stop_openuav(request):
    """Stop a running container without removing it"""
    try:
        # Get username from request data
        username = request.POST.get('username')
        
        if not username:
            messages.error(request, 'Username is required')
            return redirect('openuav_manager:manage')
        
        container_name = f'digital-twin-{username}'
        logger.info(f"Attempting to stop container: {container_name}")
        
        # Check if container exists and is running
        docker_check_cmd = f"docker ps -a --filter name={container_name} --format '{{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.ID}}}}'"
        stdout, stderr, code = run_command(docker_check_cmd)
        docker_status = stdout.strip().lower() if code == 0 else ''
        
        if not docker_status:
            logger.warning(f"Container {container_name} not found")
            messages.warning(request, 'Container not found')
            return redirect('openuav_manager:manage')
        
        if 'exited' in docker_status:
            logger.info(f"Container {container_name} is already stopped")
            # Update container status in database if needed
            try:
                container = Container.objects.get(name=container_name)
                if container.status != 'stopped':
                    container.status = 'stopped'
                    container.save()
            except Container.DoesNotExist:
                pass
            
            messages.info(request, 'Container is already stopped')
            return redirect('openuav_manager:manage')
        
        # Stop the container
        logger.info(f"Stopping container: {container_name}")
        stdout, stderr, code = run_command(f"docker stop {container_name}")
        if code != 0:
            logger.error(f"Error stopping container: {stderr}")
            messages.error(request, f'Error stopping container: {stderr}')
            return redirect('openuav_manager:manage')
            
        # Update container status in database
        try:
            container = Container.objects.get(name=container_name)
            container.status = 'stopped'
            container.save()
            logger.info(f"Updated container status to stopped in database: {container_name}")
        except Container.DoesNotExist:
            logger.warning(f"Container {container_name} not found in database")
            
        # Clean up X11 socket and VNC files since container is stopped
        try:
            run_command("rm -f /tmp/.X11-unix/X1")
            run_command("rm -f /tmp/.X1*")
            run_command("pkill -f Xvnc || true")
            run_command("pkill -f vncserver || true")
            logger.info("Cleaned up X11 and VNC files")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
            
        messages.success(request, 'Container stopped successfully')
        return redirect('openuav_manager:manage')
        
    except Exception as e:
        logger.error(f"Error in stop_openuav: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return redirect('openuav_manager:manage')

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

def check_github_fork(username):
    """Check if user has forked the RAS-SES-598 repository"""
    try:
        # GitHub API endpoint for repository forks
        api_url = f"https://api.github.com/repos/DREAMS-lab/RAS-SES-598-Space-Robotics-and-AI/forks"
        
        # Get list of forks (paginated)
        page = 1
        while True:
            response = requests.get(
                f"{api_url}?page={page}&per_page=100",
                headers={'Accept': 'application/vnd.github.v3+json'}
            )
            
            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.status_code}")
                return False
                
            forks = response.json()
            if not forks:  # No more forks to check
                break
                
            # Check if username exists in this page of forks
            for fork in forks:
                if fork['owner']['login'].lower() == username.lower():
                    return True
                    
            page += 1
            
        return False
    except Exception as e:
        logger.error(f"Error checking GitHub fork: {str(e)}")
        return False

def manage_view(request):
    """Display the OpenUAV launch form and handle container status"""
    try:
        if request.method == "POST":
            username = request.POST.get('username', '').strip()
            passcode = request.POST.get('passcode', '').strip()
            
            # Validate inputs
            if not username or not passcode:
                messages.error(request, 'Username and passcode are required')
                return render(request, 'openuav_manager/manage.html', {
                    'show_form': True
                })

            # Validate passcode
            if passcode != 'liftoff':
                messages.error(request, 'Invalid passcode')
                return render(request, 'openuav_manager/manage.html', {
                    'show_form': True
                })

            # Check if user has forked the repository
            if not check_github_fork(username):
                messages.error(request, 'Access denied: You must fork the RAS-SES-598 repository first')
                return render(request, 'openuav_manager/manage.html', {
                    'show_form': True
                })

            # Generate container name
            container_name = f'digital-twin-{username}'
            
            # Check actual Docker container status first
            docker_check_cmd = f"docker ps -a --filter name={container_name} --format '{{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.ID}}}}'"
            stdout, stderr, code = run_command(docker_check_cmd)
            docker_status = stdout.strip().lower() if code == 0 else ''
            
            # Clean up database state if Docker state doesn't match
            existing_container = Container.objects.filter(name=container_name).first()
            if existing_container and not docker_status:
                logger.warning(f"Container {container_name} exists in DB but not in Docker, cleaning up")
                existing_container.delete()
                existing_container = None
            
            if docker_status:
                # Container exists in Docker
                if 'up' in docker_status:
                    # Container is running
                    if not existing_container:
                        # Create DB record if missing
                        inspect_cmd = f"docker inspect {container_name} --format='{{{{.Id}}}}'"
                        container_id, _, _ = run_command(inspect_cmd)
                        existing_container = Container.objects.create(
                            container_id=container_id.strip(),
                            short_id=container_id.strip()[:12],
                            name=container_name,
                            status='running',
                            image='digital-twin-ras-ses-598:5.0',
                            ports={'vnc': 5901},
                            vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
                        )
                    return render(request, 'openuav_manager/manage.html', {
                        'show_form': False,
                        'status': {
                            'is_running': True,
                            'instance_count': 1
                        },
                        'container': existing_container,
                        'vnc_url': existing_container.vnc_url,
                        'username': username
                    })
                elif 'exited' in docker_status:
                    # Try to start the stopped container
                    logger.info(f"Starting stopped container: {container_name}")
                    stdout, stderr, code = run_command(f"docker start {container_name}")
                    if code == 0:
                        # Verify container is running
                        verify_cmd = f"docker ps --filter name={container_name} --format '{{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.ID}}}}'"
                        stdout, stderr, code = run_command(verify_cmd)
                        if code == 0 and 'up' in stdout.strip().lower():
                            if existing_container:
                                existing_container.status = 'running'
                                existing_container.save()
                            else:
                                # Create DB record if missing
                                inspect_cmd = f"docker inspect {container_name} --format='{{{{.Id}}}}'"
                                container_id, _, _ = run_command(inspect_cmd)
                                existing_container = Container.objects.create(
                                    container_id=container_id.strip(),
                                    short_id=container_id.strip()[:12],
                                    name=container_name,
                                    status='running',
                                    image='digital-twin-ras-ses-598:5.0',
                                    ports={'vnc': 5901},
                                    vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
                                )
                            return render(request, 'openuav_manager/manage.html', {
                                'show_form': False,
                                'status': {
                                    'is_running': True,
                                    'instance_count': 1
                                },
                                'container': existing_container,
                                'vnc_url': existing_container.vnc_url,
                                'username': username
                            })
                
                # If we get here, either start failed or container is in an unexpected state
                logger.warning(f"Container {container_name} in unexpected state or failed to start, removing")
                run_command(f"docker rm -f {container_name}")
                if existing_container:
                    existing_container.delete()

            # Clean up any existing container before launching new one
            run_command(f"docker rm -f {container_name}")
            
            # Launch new container
            launch_cmd = f"""docker run --init --runtime=nvidia --network=dreamslab --privileged --name={container_name} --hostname={container_name} -d -e VGL_DISPLAY=:0.0 -e DISPLAY=:1.0 -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw digital-twin-ras-ses-598:5.0"""
            
            # Print the full command to console
            logger.info("Launching container with command:")
            logger.info("=" * 80)
            logger.info(launch_cmd)
            logger.info("=" * 80)
            
            stdout, stderr, code = run_command(launch_cmd)
            if code != 0:
                logger.error(f"Failed to launch container: {stderr}")
                messages.error(request, f'Failed to launch container: {stderr}')
                return render(request, 'openuav_manager/manage.html', {
                    'show_form': True
                })
            
            # Get container ID from stdout
            container_id = stdout.strip()
            
            # Verify container is running
            verify_cmd = f"docker ps --filter name={container_name} --format '{{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.ID}}}}'"
            stdout, stderr, code = run_command(verify_cmd)
            if code != 0 or 'up' not in stdout.strip().lower():
                logger.error(f"Container {container_name} failed to start properly")
                run_command(f"docker rm -f {container_name}")
                messages.error(request, 'Container failed to start properly')
                return render(request, 'openuav_manager/manage.html', {
                    'show_form': True
                })
            
            # Create container record
            new_container = Container.objects.create(
                container_id=container_id,
                short_id=container_id[:12],
                name=container_name,
                status='running',
                image='digital-twin-ras-ses-598:5.0',
                ports={'vnc': 5901},
                vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
            )
            
            return render(request, 'openuav_manager/manage.html', {
                'show_form': False,
                'status': {
                    'is_running': True,
                    'instance_count': 1
                },
                'container': new_container,
                'vnc_url': new_container.vnc_url,
                'username': username
            })

        # GET request - show the initial form
        return render(request, 'openuav_manager/manage.html', {
            'show_form': True,
            'status': {
                'is_running': False,
                'instance_count': 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error in manage_view: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return render(request, 'openuav_manager/manage.html', {
            'show_form': True,
            'status': {
                'is_running': False,
                'instance_count': 0
            }
        })

@csrf_exempt
@require_http_methods(["POST"])
def reset_openuav(request):
    """Reset (remove and restart) a container instance"""
    try:
        # Get username and reset confirmation from request data
        username = request.POST.get('username')
        reset_confirm = request.POST.get('reset_confirm', '').strip().lower()
        
        if not username:
            return JsonResponse({
                'status': 'error',
                'message': 'Username is required'
            }, status=400)
        
        if reset_confirm != 'reset':
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid reset confirmation'
            }, status=400)
        
        container_name = f'digital-twin-{username}'
        logger.info(f"Resetting container: {container_name}")
        
        # Force remove the container
        stdout, stderr, code = run_command(f"docker rm -f {container_name}")
        if code != 0:
            logger.warning(f"Error removing container: {stderr}")
        
        # Remove container from database if it exists
        Container.objects.filter(name=container_name).delete()
        
      
        
        # Launch new container
        launch_cmd = f"""docker run --init --runtime=nvidia --network=dreamslab --privileged --name={container_name} --hostname={container_name} -d -e VGL_DISPLAY=:0.0 -e DISPLAY=:1.0 -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw digital-twin-ras-ses-598:5.0"""
        
        logger.info("Launching new container with command:")
        logger.info("=" * 80)
        logger.info(launch_cmd)
        logger.info("=" * 80)
        
        stdout, stderr, code = run_command(launch_cmd)
        if code != 0:
            logger.error(f"Failed to launch container: {stderr}")
            messages.error(request, f'Failed to launch container: {stderr}')
            return redirect('openuav_manager:manage')
        
        # Get container ID from stdout
        container_id = stdout.strip()
        
        # Verify container is running
        verify_cmd = f"docker ps --filter name={container_name} --format '{{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.ID}}}}'"
        stdout, stderr, code = run_command(verify_cmd)
        if code != 0 or 'up' not in stdout.strip().lower():
            logger.error(f"Container {container_name} failed to start properly")
            run_command(f"docker rm -f {container_name}")
            messages.error(request, 'Container failed to start properly')
            return redirect('openuav_manager:manage')
        
        # Create new container record
        Container.objects.create(
            container_id=container_id,
            short_id=container_id[:12],
            name=container_name,
            status='running',
            image='digital-twin-ras-ses-598:5.0',
            ports={'vnc': 5901},
            vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
        )
        
        messages.success(request, 'Container reset successfully')
        return redirect('openuav_manager:manage')
        
    except Exception as e:
        logger.error(f"Error in reset_openuav: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return redirect('openuav_manager:manage')

def swarm_view(request):
    """Display the OpenWAV swarm launch form and handle container status"""
    try:
        if request.method == "POST":
            username = request.POST.get('username', '').strip()
            passcode = request.POST.get('passcode', '').strip()
            
            # Validate inputs
            if not username or not passcode:
                messages.error(request, 'Username and passcode are required')
                return render(request, 'openuav_manager/swarm.html', {
                    'show_form': True
                })

            # Validate passcode
            if passcode != 'liftoff':
                messages.error(request, 'Invalid passcode')
                return render(request, 'openuav_manager/swarm.html', {
                    'show_form': True
                })

            # Check if user has forked the repository
            if not check_github_fork(username):
                messages.error(request, 'Access denied: You must fork the RAS-SES-598 repository first')
                return render(request, 'openuav_manager/swarm.html', {
                    'show_form': True
                })

            # Generate container name with digital-swarm prefix
            container_name = f'digital-swarm-{username}'
            
            # Check if swarm is initialized
            stdout, stderr, code = run_command("docker info --format '{{.Swarm.LocalNodeState}}'")
            if code != 0 or 'active' not in stdout.strip().lower():
                # Initialize swarm if not active
                init_cmd = "docker swarm init --advertise-addr eth0"
                stdout, stderr, code = run_command(init_cmd)
                if code != 0:
                    logger.error(f"Failed to initialize swarm: {stderr}")
                    messages.error(request, 'Failed to initialize Docker swarm')
                    return render(request, 'openuav_manager/swarm.html', {
                        'show_form': True
                    })

            # Create overlay network if it doesn't exist
            network_name = 'swarm-network'
            check_network_cmd = f"docker network ls --filter name={network_name} --format '{{{{.Name}}}}'"
            stdout, stderr, code = run_command(check_network_cmd)
            if network_name not in stdout:
                create_network_cmd = f"docker network create -d overlay {network_name}"
                stdout, stderr, code = run_command(create_network_cmd)
                if code != 0:
                    logger.error(f"Failed to create overlay network: {stderr}")
                    messages.error(request, 'Failed to create swarm network')
                    return render(request, 'openuav_manager/swarm.html', {
                        'show_form': True
                    })

            # Check if service exists
            service_check_cmd = f"docker service ls --filter name={container_name} --format '{{{{.Name}}}}'"
            stdout, stderr, code = run_command(service_check_cmd)
            
            if container_name in stdout:
                # Service exists, update it
                update_cmd = f"""docker service update --force \
                    --update-parallelism 1 \
                    --update-delay 10s \
                    {container_name}"""
                stdout, stderr, code = run_command(update_cmd)
                if code != 0:
                    logger.error(f"Failed to update service: {stderr}")
                    messages.error(request, f'Failed to update service: {stderr}')
                    return render(request, 'openuav_manager/swarm.html', {
                        'show_form': True
                    })
            else:
                # Create new service
                create_cmd = f"""docker service create \
                    --name {container_name} \
                    --network {network_name} \
                    --mount type=bind,source=/tmp/.X11-unix/X0,target=/tmp/.X11-unix/X0 \
                    -e VGL_DISPLAY=:0.0 \
                    -e DISPLAY=:1.0 \
                    --constraint 'node.role==manager' \
                    digital-twin-ras-ses-598:5.0"""
                
                stdout, stderr, code = run_command(create_cmd)
                if code != 0:
                    logger.error(f"Failed to create service: {stderr}")
                    messages.error(request, f'Failed to create service: {stderr}')
                    return render(request, 'openuav_manager/swarm.html', {
                        'show_form': True
                    })

            # Wait for service to be running
            time.sleep(5)  # Give some time for the service to start

            # Get service status
            status_cmd = f"docker service ps {container_name} --format '{{{{.CurrentState}}}}'"
            stdout, stderr, code = run_command(status_cmd)
            
            if code == 0 and 'running' in stdout.lower():
                # Create container record
                Container.objects.create(
                    container_id=container_name,  # Using service name as ID for swarm services
                    short_id=container_name[:12],
                    name=container_name,
                    status='running',
                    image='digital-twin-ras-ses-598:5.0',
                    ports={'vnc': 5901},
                    vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
                )

                return render(request, 'openuav_manager/swarm.html', {
                    'show_form': False,
                    'status': {
                        'is_running': True,
                        'instance_count': 1
                    },
                    'container': {
                        'name': container_name,
                        'vnc_url': f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1"
                    },
                    'username': username
                })
            else:
                messages.error(request, 'Service failed to start properly')
                return render(request, 'openuav_manager/swarm.html', {
                    'show_form': True
                })

        # GET request - show the initial form
        return render(request, 'openuav_manager/swarm.html', {
            'show_form': True,
            'status': {
                'is_running': False,
                'instance_count': 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error in swarm_view: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return render(request, 'openuav_manager/swarm.html', {
            'show_form': True,
            'status': {
                'is_running': False,
                'instance_count': 0
            }
        })

# ============================================================================
# ROS Manager Views
# ============================================================================

def get_rosbag_base_path():
    """Get the base path for rosbags"""
    return Path('/mnt/tesseract-store/trike-backup')


def scan_rosbags():
    """Scan for available ROS2 rosbags"""
    rosbag_base = get_rosbag_base_path()
    rosbags = []
    
    if not rosbag_base.exists():
        logger.warning(f"Rosbag base path does not exist: {rosbag_base}")
        return rosbags
    
    for rosbag_dir in rosbag_base.glob('rosbag2_*'):
        if not rosbag_dir.is_dir():
            continue
            
        metadata_file = rosbag_dir / 'metadata.yaml'
        if not metadata_file.exists():
            continue
        
        try:
            # Parse metadata.yaml
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
            
            # Get rosbag info
            bag_info = metadata.get('rosbag2_bagfile_information', {})
            topics = bag_info.get('topics_with_message_count', [])
            
            # Calculate total size
            total_size = sum(
                (rosbag_dir / f).stat().st_size 
                for f in rosbag_dir.iterdir() 
                if f.is_file()
            )
            
            # Get duration if available
            duration = bag_info.get('duration', {}).get('nanoseconds', 0) / 1e9
            
            rosbags.append({
                'path': str(rosbag_dir),
                'name': rosbag_dir.name,
                'size': total_size,
                'duration': duration,
                'topics': topics,
                'metadata': metadata,
                'topic_count': len(topics)
            })
        except Exception as e:
            logger.error(f"Error scanning rosbag {rosbag_dir}: {str(e)}")
            continue
    
    return rosbags


def get_viz_command(viz_type):
    """Get command to run visualization tool inside container"""
    commands = {
        'rviz2': 'rviz2',
        'foxglove': 'foxglove-studio',
        'plotjuggler': 'ros2 run plotjuggler plotjuggler',
        'rqt_bag': 'ros2 run rqt_bag rqt_bag',
    }
    return commands.get(viz_type, 'rviz2')


@require_http_methods(["GET"])
def rosbag_browser(request):
    """Browse available ROS2 rosbags"""
    try:
        rosbags = scan_rosbags()
        
        # Update or create database records
        for rosbag_data in rosbags:
            RosbagFile.objects.update_or_create(
                path=rosbag_data['path'],
                defaults={
                    'name': rosbag_data['name'],
                    'size': rosbag_data['size'],
                    'duration': rosbag_data['duration'],
                    'topics': rosbag_data['topics'],
                    'metadata': rosbag_data['metadata'],
                }
            )
        
        # Get from database for consistent display
        db_rosbags = RosbagFile.objects.all().order_by('-indexed')
        
        return render(request, 'openuav_manager/rosbag_browser.html', {
            'rosbags': db_rosbags,
        })
    except Exception as e:
        logger.error(f"Error in rosbag_browser: {str(e)}", exc_info=True)
        messages.error(request, f'Error browsing rosbags: {str(e)}')
        return render(request, 'openuav_manager/rosbag_browser.html', {
            'rosbags': [],
        })


@csrf_exempt
@require_http_methods(["POST"])
def launch_rosbag_viewer(request):
    """Launch ROS2 container with rosbag mounted for visualization"""
    try:
        # Get request data
        if request.POST:
            rosbag_path = request.POST.get('rosbag_path', '').strip()
            username = request.POST.get('username', request.user.username if request.user.is_authenticated else 'guest').strip()
            viz_type = request.POST.get('viz_type', 'rviz2').strip()
        else:
            try:
                data = json.loads(request.body)
                rosbag_path = data.get('rosbag_path', '').strip()
                username = data.get('username', request.user.username if request.user.is_authenticated else 'guest').strip()
                viz_type = data.get('viz_type', 'rviz2').strip()
            except json.JSONDecodeError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid request format'
                }, status=400)
        
        if not rosbag_path:
            return JsonResponse({
                'status': 'error',
                'message': 'Rosbag path is required'
            }, status=400)
        
        # Verify rosbag exists
        rosbag_path_obj = Path(rosbag_path)
        if not rosbag_path_obj.exists():
            return JsonResponse({
                'status': 'error',
                'message': f'Rosbag path does not exist: {rosbag_path}'
            }, status=404)
        
        # Get or create rosbag record
        rosbag, _ = RosbagFile.objects.get_or_create(
            path=rosbag_path,
            defaults={'name': rosbag_path_obj.name}
        )
        
        # Check if user already has an active session for this rosbag
        if request.user.is_authenticated:
            existing_session = RosbagSession.objects.filter(
                user=request.user,
                rosbag=rosbag,
                container__status='running'
            ).first()
            
            if existing_session:
                return JsonResponse({
                    'status': 'exists',
                    'message': 'Active session already exists',
                    'vnc_url': existing_session.container.vnc_url,
                    'container_id': existing_session.container.container_id
                })
        
        # Generate container name using digital-twin-ros-analyze-UID format
        # UID is the username (or 'guest' if not authenticated)
        container_name = f'digital-twin-ros-analyze-{username}'
        
        # Clean up any existing container with same name
        run_command(f"docker rm -f {container_name}")
        
        # Launch container with rosbag mounted
        launch_cmd = f"""docker run --init --runtime=nvidia --network=dreamslab --privileged \
            --name={container_name} --hostname={container_name} -d \
            -e VGL_DISPLAY=:0.0 -e DISPLAY=:1.0 \
            -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw \
            -v {rosbag_path}:/rosbags:ro \
            digital-twin-ras-ses-598:5.0 \
            /bin/bash -c "source /opt/ros/humble/setup.bash && {get_viz_command(viz_type)}"
        """
        
        logger.info("Launching ROS bag viewer container:")
        logger.info("=" * 80)
        logger.info(launch_cmd)
        logger.info("=" * 80)
        
        stdout, stderr, code = run_command(launch_cmd)
        if code != 0:
            logger.error(f"Failed to launch container: {stderr}")
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to launch container: {stderr}'
            }, status=500)
        
        container_id = stdout.strip()
        
        # Wait for container to be running
        max_retries = 5
        retry_delay = 1
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
                'status': 'error',
                'message': 'Container failed to start'
            }, status=500)
        
        # Create container record
        container = Container.objects.create(
            container_id=container_id,
            short_id=container_id[:12],
            name=container_name,
            status='running',
            image='digital-twin-ras-ses-598:5.0',
            ports={'vnc': 5901},
            vnc_url=f"https://{container_name}.deepgis.org/vnc.html?resize=remote&reconnect=1&autoconnect=1",
            user=request.user if request.user.is_authenticated else None
        )
        
        # Create rosbag session record
        session = RosbagSession.objects.create(
            container=container,
            rosbag=rosbag,
            visualization_type=viz_type,
            user=request.user if request.user.is_authenticated else None
        )
        
        return JsonResponse({
            'status': 'success',
            'message': 'ROS bag viewer launched successfully',
            'container': {
                'id': container_id,
                'short_id': container_id[:12],
                'name': container_name,
                'vnc_url': container.vnc_url
            },
            'session_id': session.id
        })
        
    except Exception as e:
        logger.error(f"Error launching rosbag viewer: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def rosbag_playback_control(request, container_id):
    """Control rosbag playback (play/pause/stop/rate)"""
    try:
        data = json.loads(request.body) if not request.POST else request.POST
        action = data.get('action', '').strip()
        rate = float(data.get('rate', 1.0))
        loop = data.get('loop', 'false').lower() == 'true'
        
        # Get container and session
        try:
            container = Container.objects.get(container_id=container_id, status='running')
            session = RosbagSession.objects.get(container=container)
        except (Container.DoesNotExist, RosbagSession.DoesNotExist):
            return JsonResponse({
                'status': 'error',
                'message': 'Container or session not found'
            }, status=404)
        
        rosbag_name = Path(session.rosbag.path).name
        loop_flag = '--loop' if loop else ''
        
        if action == 'play':
            # Play rosbag
            play_cmd = f"""docker exec {container_id} /bin/bash -c \
                "source /opt/ros/humble/setup.bash && \
                 ros2 bag play /rosbags/{rosbag_name} --rate {rate} {loop_flag} &"
            """
            stdout, stderr, code = run_command(play_cmd)
            if code == 0:
                session.is_playing = True
                session.playback_rate = rate
                session.is_looping = loop
                session.save()
                return JsonResponse({'status': 'success', 'message': 'Playing rosbag'})
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Failed to play: {stderr}'
                }, status=500)
        
        elif action == 'stop':
            # Stop rosbag playback
            stop_cmd = f"""docker exec {container_id} /bin/bash -c \
                "pkill -f 'ros2 bag play' || true"
            """
            stdout, stderr, code = run_command(stop_cmd)
            if code == 0:
                session.is_playing = False
                session.save()
                return JsonResponse({'status': 'success', 'message': 'Stopped rosbag'})
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Failed to stop: {stderr}'
                }, status=500)
        
        else:
            return JsonResponse({
                'status': 'error',
                'message': f'Invalid action: {action}'
            }, status=400)
            
    except Exception as e:
        logger.error(f"Error in rosbag_playback_control: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def rosbag_sessions(request):
    """List active rosbag visualization sessions"""
    try:
        if request.user.is_authenticated:
            sessions = RosbagSession.objects.filter(
                user=request.user,
                container__status='running'
            ).select_related('container', 'rosbag').order_by('-created')
        else:
            sessions = []
        
        return render(request, 'openuav_manager/rosbag_sessions.html', {
            'sessions': sessions,
        })
    except Exception as e:
        logger.error(f"Error in rosbag_sessions: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return render(request, 'openuav_manager/rosbag_sessions.html', {
            'sessions': [],
        })
