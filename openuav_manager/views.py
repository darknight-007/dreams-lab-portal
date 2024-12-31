from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import subprocess
import json
from datetime import datetime
from .models import Container
import logging
import uuid
import psutil
from django.conf import settings

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
    """Sync container states with database"""
    try:
        # Get all running containers
        stdout, stderr, code = run_command("docker ps -a --format '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}'")
        if code != 0:
            logger.error(f"Error getting container list: {stderr}")
            return
            
        # Parse docker ps output
        current_containers = {}
        for line in stdout.splitlines():
            if line.strip():
                try:
                    container_id, name, status, image, ports = line.split('\t')
                    # Parse ports information
                    port_dict = {}
                    if ports:
                        port_mappings = ports.split(',')
                        for mapping in port_mappings:
                            if '->' in mapping:
                                host_port = mapping.split('->')[0].strip().split(':')[-1]
                                container_port = mapping.split('->')[1].strip().split('/')[0]
                                port_dict[host_port] = container_port
                    
                    current_containers[container_id] = {
                        'name': name,
                        'status': 'running' if 'Up' in status else 'stopped',
                        'image': image,
                        'ports': port_dict
                    }
                except ValueError as e:
                    logger.error(f"Error parsing container info: {str(e)}")
                    continue
        
        # Update database records
        db_containers = Container.objects.all()
        
        # Update existing containers
        for container in db_containers:
            if container.container_id in current_containers:
                info = current_containers[container.container_id]
                container.status = info['status']
                container.ports = info['ports']
                container.save()
                del current_containers[container.container_id]
            else:
                # Container no longer exists
                container.delete()
        
        # Add new containers that match our naming pattern
        for container_id, info in current_containers.items():
            if info['name'].startswith('openuav-'):
                Container.objects.create(
                    container_id=container_id,
                    unique_id=str(uuid.uuid4()),
                    name=info['name'],
                    status=info['status'],
                    created=datetime.now(),
                    ports=info['ports'],
                    image=info['image']
                )
                
    except Exception as e:
        logger.error(f"Error syncing containers: {str(e)}", exc_info=True)

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
            stdout, stderr, code = run_command(f"docker stop {container_id}")
            if code == 0:
                messages.success(request, f'Container {container_id} stopped successfully')
            else:
                messages.error(request, f'Error stopping container: {stderr}')
        elif action == 'delete':
            # First stop the container if it's running
            run_command(f"docker stop {container_id}")
            # Then remove it
            stdout, stderr, code = run_command(f"docker rm {container_id}")
            if code == 0:
                messages.success(request, f'Container {container_id} deleted successfully')
                # Remove from database
                Container.objects.filter(container_id=container_id).delete()
            else:
                messages.error(request, f'Error deleting container: {stderr}')
        
        # Allow time for container status to update
        import time
        time.sleep(1)
        sync_containers()
        
    except Exception as e:
        logger.error(f"Error in container_action: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success'})
    return redirect('container_list')

@csrf_exempt
@require_http_methods(["GET", "POST"])
def launch_openuav(request):
    logger.info("Received request to launch OpenUAV")
    try:
        # Check for existing active containers for this session
        session_filter = {
            'session_type': 'guest',  # For now, assume guest session
            'status': 'running'
        }
        if request.user.is_authenticated:
            session_filter['user'] = request.user
        
        existing_container = Container.objects.filter(**session_filter).first()
        if existing_container:
            logger.info(f"Found existing container: {existing_container.name}")
            return JsonResponse({
                'status': 'success',
                'message': 'Existing OpenUAV instance found',
                'vnc_url': 'http://deepgis.org:6080/vnc.html?resize=remote&reconnect=1&autoconnect=1',
                'container_name': existing_container.name,
                'unique_id': existing_container.unique_id
            })

        # Generate a unique ID for this container
        unique_id = str(uuid.uuid4())
        container_name = f"openuav-{unique_id[:8]}"
        
        logger.info(f"Launching container with name: {container_name}")
        
        # First check if container exists
        logger.info("Checking if container already exists...")
        stdout, stderr, code = run_command(f"docker ps -a --filter name={container_name} --format '{{{{.Names}}}}'")
        
        if code != 0:
            logger.error(f"Error checking container existence: {stderr}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error checking container: {stderr}'
            }, status=500)
            
        # If container exists, stop and remove it
        if container_name in stdout:
            logger.info("Container exists, cleaning up...")
            run_command(f"docker stop {container_name}")
            run_command(f"docker rm {container_name}")
            
        # Thorough cleanup of any stale files
        logger.info("Cleaning up stale files...")
        run_command("rm -f /tmp/.X11-unix/X1")
        run_command("rm -f /tmp/.X1*")
        run_command("pkill -f Xvnc || true")
        run_command("pkill -f vncserver || true")
            
        # Create new container
        logger.info("Creating new container...")
        cmd = (f"docker run -d --name {container_name} --privileged "
              "--gpus all "
              "-p 6080:6080 -p 5901:5901 "
              "-e NVIDIA_VISIBLE_DEVICES=all "
              "-e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute "
              "-e VNC_PASSWORD=liftoff "
              "-v /tmp/.X11-unix:/tmp/.X11-unix:rw "
              "--runtime=nvidia "
              "openuav:px4-sitl "
              "/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf")
        stdout, stderr, code = run_command(cmd)
        
        if code == 0:
            logger.info("Container started successfully")
            # Wait for services to initialize
            import time
            time.sleep(5)  # Initial wait
            
            # Get container ID
            container_id_stdout, _, _ = run_command(f"docker inspect --format='{{{{.Id}}}}' {container_name}")
            container_id = container_id_stdout.strip()
            
            # Create or update container record in database
            container = Container.objects.create(
                container_id=container_id,
                unique_id=unique_id,
                name=container_name,
                status='running',
                created=datetime.now(),
                ports={'6080': '6080', '5901': '5901'},
                image='openuav:px4-sitl',
                session_type='guest',
                user=request.user if request.user.is_authenticated else None
            )
            
            # Verify container is running
            verify_stdout, verify_stderr, verify_code = run_command(f"docker inspect {container_name} --format '{{{{.State.Status}}}} {{{{.State.Running}}}}'")
            logger.info(f"Container state: {verify_stdout}")
            
            # Check if TurboVNC is running
            vnc_check, vnc_stderr, vnc_code = run_command(f"docker exec {container_name} ps aux | grep Xvnc")
            if 'Xvnc' not in vnc_check:
                logger.error("TurboVNC not running, attempting to start it...")
                # Clean up any stale files inside the container
                run_command(f"docker exec {container_name} rm -f /tmp/.X11-unix/X1 /tmp/.X1* /root/.vnc/*.pid /root/.vnc/*.log")
                time.sleep(1)  # Wait for cleanup
                
                # Try to start TurboVNC manually
                run_command(f"docker exec {container_name} /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24")
                time.sleep(2)  # Wait for VNC server to start
                
                # Verify TurboVNC again
                vnc_check, _, _ = run_command(f"docker exec {container_name} ps aux | grep Xvnc")
                if 'Xvnc' not in vnc_check:
                    logger.error("Failed to start TurboVNC")
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to start TurboVNC server'
                    }, status=500)
            
            # Get container logs for debugging
            logs_stdout, logs_stderr, logs_code = run_command(f"docker logs {container_name} --tail 20")
            logger.info(f"Container recent logs: {logs_stdout}")
            if logs_stderr:
                logger.warning(f"Container logs errors: {logs_stderr}")
            
            return JsonResponse({
                'status': 'success',
                'message': 'OpenUAV instance launched successfully',
                'vnc_url': 'http://deepgis.org:6080/vnc.html?resize=remote&reconnect=1&autoconnect=1',
                'container_state': verify_stdout,
                'container_name': container_name,
                'unique_id': unique_id
            })
        else:
            logger.error(f"Failed to start container: {stderr}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error launching container: {stderr}'
            }, status=500)
    except Exception as e:
        logger.error(f"Exception in launch_openuav: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
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

def manage_openuav(request):
    """Render the OpenUAV management interface"""
    # Get current status
    stdout, stderr, code = run_command("docker ps --format '{{json .}}' --filter name=openuav")
    
    status = {
        'is_running': False,
        'instance_count': 0
    }
    
    if code == 0:
        containers = [json.loads(line) for line in stdout.split('\n') if line]
        if containers:
            status = {
                'is_running': True,
                'instance_count': 1
            }
    
    return render(request, 'openuav_manager/manage.html', {
        'status': status,
        'vnc_url': 'http://deepgis.org:6080/vnc.html?resize=remote&reconnect=1&autoconnect=1'
    })

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
        
        if not action or not container_ids:
            return JsonResponse({
                'status': 'error',
                'message': 'Missing action or container IDs'
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
