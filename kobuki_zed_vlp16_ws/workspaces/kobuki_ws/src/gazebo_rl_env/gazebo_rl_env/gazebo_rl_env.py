import rclpy
from rclpy.node import Node

from std_srvs.srv import Empty

import math
import time
import numpy as np


class GAZEBO_RL_ENV_NODE(Node):
    def __init__(self):
        super().__init__("gazebo_rl_env_node")

        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.reset_world_client = self.create_client(Empty, "/reset_world")

        self.current_timestamp = 0

        self.config = {
            "step_time_delta": 0.05, # seconds
            "gazebo_service_timeout": 1.0, # seconds
        }

    def reset(self):
        self.current_timestamp = 0

        while not self.reset_world_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "reset_world" not available, waiting again...')
        self.reset_world_client.call_async(Empty.Request())

    def step(self):
        self.current_timestamp += 1
        # self.get_logger().info(f"Current timestamp: {self.current_timestamp}")

        # Unpause the physics to allow the simulation to run
        while not self.unpause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "unpause" not available, waiting again...')
        self.unpause_client.call_async(Empty.Request())

        # Wait for the Gazebo to run for a certain amount of time.
        # Note that Gazebo will run 1000 iterations per second by default,
        # so the real simulation timestamp will be: 0.001 * self.config["step_time_delta"] * current_timestamp
        time.sleep(self.config["step_time_delta"])

        # Pause the physics to stop the simulation
        while not self.pause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "pause" not available, waiting again...')
        self.pause_client.call_async(Empty.Request())
