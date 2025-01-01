#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleCommand, VehicleLocalPosition, VehicleStatus
from functools import partial
import json
import asyncio

class PX4QuizBridge(Node):
    def __init__(self):
        super().__init__('px4_quiz_bridge')
        
        # Parameters
        self.declare_parameter('quiz_id', '')
        self.declare_parameter('student_id', '')
        
        # State tracking
        self.current_position = None
        self.current_status = None
        self.quiz_active = False
        
        # Publishers
        self.position_pub = self.create_publisher(
            PoseStamped,
            '/fmu/in/setpoint_position',
            10
        )
        
        # Subscribers
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            10
        )
        
        # Services
        self.param_client = self.create_client(
            SetParameters,
            '/fmu/parameters/set'
        )
        
        # Create services for quiz interaction
        self.create_service(
            SetParameters,
            'quiz/set_parameters',
            self.handle_quiz_parameters
        )
        
        # Timer for state publishing
        self.create_timer(0.1, self.publish_state)
        
        self.get_logger().info('PX4 Quiz Bridge initialized')
    
    def position_callback(self, msg):
        """Store current position"""
        self.current_position = msg
    
    def status_callback(self, msg):
        """Store vehicle status"""
        self.current_status = msg
    
    def publish_state(self):
        """Publish current state for quiz validation"""
        if not self.quiz_active:
            return
            
        if self.current_position and self.current_status:
            state = {
                'position': {
                    'x': self.current_position.x,
                    'y': self.current_position.y,
                    'z': self.current_position.z
                },
                'status': {
                    'nav_state': self.current_status.nav_state,
                    'arming_state': self.current_status.arming_state
                },
                'quiz_id': self.get_parameter('quiz_id').value,
                'student_id': self.get_parameter('student_id').value
            }
            
            # TODO: Publish state to quiz validation service
            self.get_logger().info(f'Quiz state: {json.dumps(state)}')
    
    async def handle_quiz_parameters(self, request, response):
        """Handle parameter changes from quiz"""
        results = []
        for param in request.parameters:
            try:
                # Verify parameter is allowed for quiz
                if not self.validate_quiz_parameter(param):
                    results.append(self.create_parameter_result(
                        param.name,
                        False,
                        'Parameter not allowed in quiz'
                    ))
                    continue
                
                # Set parameter on PX4
                future = self.param_client.call_async(SetParameters.Request(
                    parameters=[param]
                ))
                
                # Wait for response
                await future
                result = future.result()
                
                results.append(result.results[0])
                
            except Exception as e:
                results.append(self.create_parameter_result(
                    param.name,
                    False,
                    str(e)
                ))
        
        response.results = results
        return response
    
    def validate_quiz_parameter(self, param):
        """Verify parameter is allowed to be changed in quiz"""
        # TODO: Implement parameter validation rules
        allowed_params = [
            'MPC_XY_VEL_MAX',
            'MPC_Z_VEL_MAX',
            'MPC_TILTMAX_AIR'
        ]
        return param.name in allowed_params
    
    def create_parameter_result(self, name, success, message):
        """Create a parameter result message"""
        result = rcl_interfaces.msg.SetParametersResult()
        result.successful = success
        result.reason = message
        return result

def main(args=None):
    rclpy.init(args=args)
    node = PX4QuizBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 