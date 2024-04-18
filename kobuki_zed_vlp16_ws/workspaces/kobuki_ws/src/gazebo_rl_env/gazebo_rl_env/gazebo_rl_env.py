import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty

from ament_index_python.packages import get_package_share_directory

import os
import math
import time
import numpy as np


class GAZEBO_RL_ENV_NODE(Node):
    def __init__(self):
        super().__init__("gazebo_rl_env_node")
        
        self.config = {
            "step_time_delta": 0.05,  # seconds
            "gazebo_service_timeout": 1.0,  # seconds
        }
        
        self.current_timestamp = 0

        # Create the clients for the Gazebo services
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.reset_world_client = self.create_client(Empty, "/reset_world")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")

        # Load the ball URDF
        ball_urdf_path = os.path.join(get_package_share_directory("gazebo_rl_env"), "urdf", "ball.urdf")
        self.ball_urdf = open(ball_urdf_path, "r").read()

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

    def spawn_ball(self, x: float, y: float, z: float, name: str):
        self.spawn_ball_request = SpawnEntity.Request()
        self.spawn_ball_request.name = name
        self.spawn_ball_request.xml = self.ball_urdf
        self.spawn_ball_request.initial_pose.position.x = x
        self.spawn_ball_request.initial_pose.position.y = y
        self.spawn_ball_request.initial_pose.position.z = z

        # Spawn the ball
        while not self.spawn_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "spawn_entity" not available, waiting again...')
        self.spawn_entity_client.call_async(self.spawn_ball_request)

    def delete_ball(self, name: str):
        self.delete_ball_request = DeleteEntity.Request()
        self.delete_ball_request.name = name

        # Delete the ball
        while not self.delete_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "delete_entity" not available, waiting again...')
        self.delete_entity_client.call_async(self.delete_ball_request)
