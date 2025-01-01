#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PoseStamped
import json
import numpy as np
from typing import Dict, List, Any

class QuizValidator(Node):
    def __init__(self):
        super().__init__('quiz_validator')
        
        # Quiz definitions
        self.quizzes: Dict[str, Dict[str, Any]] = {
            'hover_test': {
                'type': 'position',
                'target': [0, 0, 2],
                'tolerance': 0.1,
                'time_requirement': 5.0,  # seconds
                'parameters': {
                    'MPC_XY_VEL_MAX': {
                        'min': 0.5,
                        'max': 5.0
                    }
                }
            },
            'square_trajectory': {
                'type': 'trajectory',
                'points': [
                    [0, 0, 2],
                    [5, 0, 2],
                    [5, 5, 2],
                    [0, 5, 2],
                    [0, 0, 2]
                ],
                'point_tolerance': 0.2,
                'time_per_point': 10.0
            }
        }
        
        # Active quiz tracking
        self.active_quiz = None
        self.quiz_start_time = None
        self.position_history = []
        
        # Create services
        self.create_service(
            SetParameters,
            'quiz/start',
            self.handle_start_quiz
        )
        
        self.create_service(
            SetParameters,
            'quiz/validate',
            self.handle_validate_quiz
        )
        
        # Subscribe to vehicle state
        self.create_subscription(
            PoseStamped,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            10
        )
        
        self.get_logger().info('Quiz Validator initialized')
    
    def position_callback(self, msg):
        """Record position for trajectory validation"""
        if not self.active_quiz:
            return
            
        self.position_history.append({
            'timestamp': self.get_clock().now().to_msg(),
            'position': [msg.pose.position.x, 
                        msg.pose.position.y, 
                        msg.pose.position.z]
        })
    
    def handle_start_quiz(self, request, response):
        """Start a new quiz attempt"""
        try:
            quiz_id = request.parameters[0].string_value
            if quiz_id not in self.quizzes:
                response.results = [self.create_result(False, 'Invalid quiz ID')]
                return response
            
            self.active_quiz = self.quizzes[quiz_id]
            self.quiz_start_time = self.get_clock().now()
            self.position_history = []
            
            response.results = [self.create_result(True, 'Quiz started')]
            
        except Exception as e:
            response.results = [self.create_result(False, str(e))]
        
        return response
    
    def handle_validate_quiz(self, request, response):
        """Validate current quiz attempt"""
        if not self.active_quiz:
            response.results = [self.create_result(False, 'No active quiz')]
            return response
        
        try:
            if self.active_quiz['type'] == 'position':
                result = self.validate_position_quiz()
            elif self.active_quiz['type'] == 'trajectory':
                result = self.validate_trajectory_quiz()
            else:
                result = (False, 'Unknown quiz type')
            
            response.results = [self.create_result(result[0], result[1])]
            
        except Exception as e:
            response.results = [self.create_result(False, str(e))]
        
        return response
    
    def validate_position_quiz(self) -> tuple[bool, str]:
        """Validate position holding quiz"""
        if not self.position_history:
            return False, 'No position data recorded'
        
        # Get recent positions
        recent_positions = self.position_history[-50:]  # Last 5 seconds at 10Hz
        
        # Check if drone is near target
        target = np.array(self.active_quiz['target'])
        tolerance = self.active_quiz['tolerance']
        
        for pos_data in recent_positions:
            pos = np.array(pos_data['position'])
            if np.linalg.norm(pos - target) > tolerance:
                return False, 'Position not maintained within tolerance'
        
        return True, 'Position maintained successfully'
    
    def validate_trajectory_quiz(self) -> tuple[bool, str]:
        """Validate trajectory following quiz"""
        if not self.position_history:
            return False, 'No position data recorded'
        
        points = self.active_quiz['points']
        tolerance = self.active_quiz['point_tolerance']
        time_per_point = self.active_quiz['time_per_point']
        
        # Check if each point was reached
        current_point = 0
        for pos_data in self.position_history:
            pos = np.array(pos_data['position'])
            target = np.array(points[current_point])
            
            if np.linalg.norm(pos - target) <= tolerance:
                current_point += 1
                if current_point >= len(points):
                    break
        
        if current_point < len(points):
            return False, f'Only reached {current_point} of {len(points)} points'
        
        return True, 'Trajectory completed successfully'
    
    def create_result(self, success: bool, message: str):
        """Create a parameter result message"""
        result = rcl_interfaces.msg.SetParametersResult()
        result.successful = success
        result.reason = message
        return result

def main(args=None):
    rclpy.init(args=args)
    node = QuizValidator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 