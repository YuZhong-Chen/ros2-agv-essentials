import rclpy
from rclpy.node import Node

from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from cv_bridge import CvBridge

import os
import math
import time
import numpy as np


class RL_ENV(Node):
    def __init__(self):
        super().__init__("rl_env")

        self.config = {
            "service_timeout": 1.0,  # seconds
        }

        # Create a publisher for the action
        self.action_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Create a subscriber for the observation
        self.observation = Image()
        self.observation_subscriber = self.create_subscription(Image, "/zed/zed_node/left/image_rect_color", self.observation_callback, 10)

        # Create a subscriber for the env info
        self.env_info_subscriber = self.create_subscription(Float32MultiArray, "/rl_env/info", self.env_info_callback, 10)

        # Create a service client for controlling the env
        self.step_service = self.create_client(Empty, "/rl_env/step")
        self.reset_service = self.create_client(Empty, "/rl_env/reset")

        # OpenCV bridge
        # Reference: https://wiki.ros.org/cv_bridge
        self.cv_bridge = CvBridge()

        # Action space
        # 0 - stop
        # 1 - forward
        # 2 - forward right
        # 3 - forward left
        # 5 - turn right
        # 6 - turn left
        self.action_space = 6

        self.timestamp = 0
        self.reward = 0

    def observation_callback(self, msg):
        self.observation = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def env_info_callback(self, msg):
        self.timestamp = msg.data[0]
        self.reward = msg.data[1]

    def publish_action(self, action: int):
        twist = Twist()

        if action == 0:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif action == 1:
            twist.linear.x = 0.5
            twist.angular.z = 0.0
        elif action == 2:
            twist.linear.x = 0.5
            twist.angular.z = -0.5
        elif action == 3:
            twist.linear.x = 0.5
            twist.angular.z = 0.5
        elif action == 4:
            twist.linear.x = 0.0
            twist.angular.z = -0.5
        elif action == 5:
            twist.linear.x = 0.0
            twist.angular.z = 0.5

        self.action_publisher.publish(twist)

    def reset(self):
        # Call the reset service
        while not self.reset_service.wait_for_service(timeout_sec=self.config["service_timeout"]):
            self.get_logger().info('Env service "reset" not available, waiting again...')
        
        future = self.reset_service.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

    def step(self, action: int):
        self.publish_action(action)

        # Call the step service
        while not self.step_service.wait_for_service(timeout_sec=self.config["service_timeout"]):
            self.get_logger().info('Env service "step" not available, waiting again...')
        
        future = self.step_service.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        return self.observation, self.reward, False
