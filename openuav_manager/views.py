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
    """Sync containers from Docker daemon to database"""
    stdout, stderr, code = run_command("docker ps -a --format '{{json .}}'")
    if code != 0:
        print(f"Error getting containers: {stderr}")
        return
    
    # Parse container info
    containers = [json.loads(line) for line in stdout.split('\n') if line]
    
    # Update database
    for container in containers:
        Container.objects.update_or_create(
            container_id=container['ID'],
            defaults={
                'name': container['Names'],
                'status': container['Status'],
                'created': datetime.strptime(container['CreatedAt'], '%Y-%m-%d %H:%M:%S %z'),
                'ports': container['Ports'],
                'image': container['Image']
            }
        )
    
    # Remove containers that no longer exist
    Container.objects.exclude(container_id__in=[c['ID'] for c in containers]).delete()

def container_list(request):
    """Display list of containers and their status"""
    sync_containers()
    containers = Container.objects.all().order_by('-created')
    return render(request, 'openuav_manager/container_list.html', {'containers': containers})

def container_action(request, container_id, action):
    """Handle container actions (start/stop)"""
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
        
        # Allow time for container status to update
        import time
        time.sleep(1)
        sync_containers()
        
    except Exception as e:
        messages.error(request, f'Error: {str(e)}')
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success'})
    return redirect('container_list')

@csrf_exempt
@require_http_methods(["GET", "POST"])
def launch_openuav(request):
    logger.info("Received request to launch OpenUAV")
    try:
        # First check if container exists
        logger.info("Checking if container already exists...")
        stdout, stderr, code = run_command("docker ps -a --filter name=openuav --format '{{.Names}}'")
        
        if code != 0:
            logger.error(f"Error checking container existence: {stderr}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error checking container: {stderr}'
            }, status=500)
            
        # If container exists, stop and remove it
        if 'openuav' in stdout:
            logger.info("Container exists, cleaning up...")
            run_command("docker stop openuav")
            run_command("docker rm openuav")
            
        # Thorough cleanup of any stale files
        logger.info("Cleaning up stale files...")
        run_command("rm -f /tmp/.X11-unix/X1")
        run_command("rm -f /tmp/.X1*")
        run_command("pkill -f Xvnc || true")
        run_command("pkill -f vncserver || true")
            
        # Create new container
        logger.info("Creating new container...")
        cmd = ("docker run -d --name openuav --privileged "
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
            
            # Verify container is running
            verify_stdout, verify_stderr, verify_code = run_command("docker inspect openuav --format '{{.State.Status}} {{.State.Running}}'")
            logger.info(f"Container state: {verify_stdout}")
            
            # Check if TurboVNC is running
            vnc_check, vnc_stderr, vnc_code = run_command("docker exec openuav ps aux | grep Xvnc")
            if 'Xvnc' not in vnc_check:
                logger.error("TurboVNC not running, attempting to start it...")
                # Clean up any stale files inside the container
                run_command("docker exec openuav rm -f /tmp/.X11-unix/X1 /tmp/.X1* /root/.vnc/*.pid /root/.vnc/*.log")
                time.sleep(1)  # Wait for cleanup
                
                # Try to start TurboVNC manually
                run_command("docker exec openuav /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24")
                time.sleep(2)  # Wait for VNC server to start
                
                # Verify TurboVNC again
                vnc_check, _, _ = run_command("docker exec openuav ps aux | grep Xvnc")
                if 'Xvnc' not in vnc_check:
                    logger.error("Failed to start TurboVNC")
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to start TurboVNC server'
                    }, status=500)
            
            # Get container logs for debugging
            logs_stdout, logs_stderr, logs_code = run_command("docker logs openuav --tail 20")
            logger.info(f"Container recent logs: {logs_stdout}")
            if logs_stderr:
                logger.warning(f"Container logs errors: {logs_stderr}")
            
            return JsonResponse({
                'status': 'success',
                'message': 'OpenUAV instance launched successfully',
                'vnc_url': 'http://deepgis.org:6080/vnc.html?resize=remote&reconnect=1&autoconnect=1',
                'container_state': verify_stdout
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
        # First stop the container
        stdout, stderr, code = run_command("docker stop openuav")
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error stopping container: {stderr}'
            }, status=500)
            
        # Then remove the container
        stdout, stderr, code = run_command("docker rm openuav")
        if code != 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Error removing container: {stderr}'
            }, status=500)
            
        # Clean up X11 socket
        run_command("rm -f /tmp/.X11-unix/X1")
            
        return JsonResponse({
            'status': 'success',
            'message': 'OpenUAV instance stopped and removed successfully'
        })
    except Exception as e:
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
