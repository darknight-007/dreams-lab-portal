#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PoseStamped
import time

class QuizTester(Node):
    def __init__(self):
        super().__init__('quiz_tester')
        
        # Create clients for quiz services
        self.start_client = self.create_client(
            SetParameters,
            'quiz/start'
        )
        self.validate_client = self.create_client(
            SetParameters,
            'quiz/validate'
        )
        
        # Create publisher for position setpoints
        self.position_pub = self.create_publisher(
            PoseStamped,
            '/fmu/in/setpoint_position',
            10
        )
        
        self.get_logger().info('Quiz Tester initialized')
    
    async def run_hover_test(self):
        """Run a simple hover test"""
        # Start the quiz
        req = SetParameters.Request()
        param = Parameter()
        param.name = 'quiz_id'
        param.value.string_value = 'hover_test'
        req.parameters = [param]
        
        self.get_logger().info('Starting hover test...')
        response = await self.start_client.call_async(req)
        if not response.results[0].successful:
            self.get_logger().error(f'Failed to start quiz: {response.results[0].reason}')
            return
        
        # Send hover setpoint
        setpoint = PoseStamped()
        setpoint.pose.position.x = 0.0
        setpoint.pose.position.y = 0.0
        setpoint.pose.position.z = 2.0  # 2 meters height
        
        self.get_logger().info('Publishing hover setpoint...')
        for _ in range(50):  # Publish for 5 seconds
            self.position_pub.publish(setpoint)
            time.sleep(0.1)
        
        # Validate quiz
        self.get_logger().info('Validating quiz...')
        response = await self.validate_client.call_async(SetParameters.Request())
        self.get_logger().info(f'Quiz result: {response.results[0].reason}')

def main(args=None):
    rclpy.init(args=args)
    tester = QuizTester()
    
    # Run hover test
    rclpy.spin_until_future_complete(
        tester,
        tester.run_hover_test()
    )
    
    rclpy.shutdown()

if __name__ == '__main__':
    main() 