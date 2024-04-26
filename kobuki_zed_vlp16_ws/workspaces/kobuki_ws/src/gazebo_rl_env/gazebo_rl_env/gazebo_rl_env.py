import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty

from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ament_index_python.packages import get_package_share_directory

import os
import math
import time
import numpy as np


class RESET_SERVICE(Node):
    def __init__(self):
        super().__init__("reset_service")

        self.reset_service = self.create_service(Empty, "/rl_env/reset", self.reset_callback)
        self.is_reset = False

    def reset_callback(self, request, response):
        self.is_reset = True

        while self.is_reset:
            # Wait for the reset to finish
            # Note that the reset will be finished in the main thread
            time.sleep(0.05)

        return response


class STEP_SERVICE(Node):
    def __init__(self):
        super().__init__("step_service")

        self.step_service = self.create_service(Empty, "/rl_env/step", self.step_callback)
        self.is_step = False

    def step_callback(self, request, response):
        self.is_step = True

        while self.is_step:
            # Wait for the step to finish
            # Note that the step will be finished in the main thread
            time.sleep(0.05)

        return response


class GAZEBO_RL_ENV_NODE(Node):
    def __init__(self):
        super().__init__("gazebo_rl_env_node")

        self.config = {
            "step_time_delta": 0.5,  # seconds
            "gazebo_service_timeout": 1.0,  # seconds
            "reach_target_distance": 0.2,  # meters
            "target_reward": 100,
            "penalty_per_step": -0.01,
        }

        self.current_timestamp = 0
        self.current_reward = 0.0

        # Create the clients for the Gazebo services
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.reset_world_client = self.create_client(Empty, "/reset_world")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.get_entity_state_client = self.create_client(GetEntityState, "/get_entity_state")
        self.set_entity_state_client = self.create_client(SetEntityState, "/set_entity_state")

        # Create the publisher for the environment
        self.info_publisher = self.create_publisher(Float32MultiArray, "/rl_env/info", 10)

        # Load the ball URDF
        ball_urdf_path = os.path.join(get_package_share_directory("gazebo_rl_env"), "urdf", "ball.urdf")
        self.ball_urdf = open(ball_urdf_path, "r").read()

        self.target_list = [None for _ in range(10)]
        
        self.reset()

    def reset(self):
        self.current_timestamp = 0
        self.current_reward = 0.0
        
        self.delete_target()

        while not self.reset_world_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "reset_world" not available, waiting again...')
        future = self.reset_world_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        # Wait for the world stable
        time.sleep(1.0)

        # Pause the physics to stop the simulation at the beginning
        while not self.pause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "pause" not available, waiting again...')
        future = self.pause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])
        
        self.generate_target()

        self.get_logger().info("Reset environment.")

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

        # Reset the reward
        self.current_reward = self.config["penalty_per_step"]

        # Check whether the Kobuki reaches the target
        state = self.get_kobuki_state()
        for i in range(10):
            if self.target_list[i] is not None:
                distance = self.target_list[i].get_distance(state)
                if distance < self.config["reach_target_distance"]:
                    self.get_logger().info(f"Kobuki reaches target {self.target_list[i].name} at timestamp {self.current_timestamp}")
                    self.delete_ball(self.target_list[i].name)
                    self.current_reward = self.config["target_reward"]
                    self.target_list[i] = None

        # Publish the information
        self.publish_info()

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
        future = self.spawn_entity_client.call_async(self.spawn_ball_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

    def delete_ball(self, name: str):
        self.delete_ball_request = DeleteEntity.Request()
        self.delete_ball_request.name = name

        # Delete the ball
        while not self.delete_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "delete_entity" not available, waiting again...')
        self.delete_entity_client.call_async(self.delete_ball_request)

    def spawn_kobuki(self, x: float, y: float, z: float, yaw: float):
        # Use system call to spawn the Kobuki robot
        os.system(f"ros2 run gazebo_ros spawn_entity.py -entity kobuki -topic /robot_description -x {x} -y {y} -z {z} -Y {yaw}")

    def delete_kobuki(self):
        self.delete_kobuki_request = DeleteEntity.Request()
        self.delete_kobuki_request.name = "kobuki"

        # Delete the Kobuki
        while not self.delete_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "delete_entity" not available, waiting again...')
        self.delete_entity_client.call_async(self.delete_kobuki_request)

    def set_entity_state(self, name: str, x: float, y: float, z: float):
        self.set_entity_state_request = SetEntityState.Request()
        self.set_entity_state_request.state.name = name
        self.set_entity_state_request.state.pose.position.x = x
        self.set_entity_state_request.state.pose.position.y = y
        self.set_entity_state_request.state.pose.position.z = z

        # Set the entity state
        while not self.set_entity_state_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "set_entity_state" not available, waiting again...')
        future = self.set_entity_state_client.call_async(self.set_entity_state_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

    def get_kobuki_state(self):
        self.get_kobuki_state_request = GetEntityState.Request()
        self.get_kobuki_state_request.name = "kobuki"

        while not self.get_entity_state_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "get_entity_state" not available, waiting again...')

        response = None
        while response is None:
            future = self.get_entity_state_client.call_async(self.get_kobuki_state_request)

            # Get the response
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])
            response = future.result()

            if response is None:
                self.get_logger().info('Gazebo service "get_entity_state" call failed, waiting again...')

        # Print the state
        # kobuki_pose = response.state.pose
        # self.get_logger().info(f"Kobuki position: ({kobuki_pose.position.x}, {kobuki_pose.position.y}, {kobuki_pose.position.z})")

        return response.state

    def generate_target(self):
        # Generate 10 targets with random y coordinates, and fixed x and z coordinates
        for i in range(10):
            x = 1.0 * (i + 1)
            y = np.random.uniform(-1.0, 1.0)
            z = 0.2
            name = "target_" + str(i)

            if self.target_list[i] is None:
                self.spawn_ball(x, y, z, name)
                self.target_list[i] = TARGET(x, y, z, name)
            else:
                self.set_entity_state(name, x, y, z)
                self.target_list[i].x = x
                self.target_list[i].y = y
                self.target_list[i].z = z
                self.get_logger().info(f"Target {name} is reset to position ({x}, {y}, {z})")
    
    def delete_target(self):
        for i in range(10):
            if self.target_list[i] is not None:
                self.delete_ball(self.target_list[i].name)
                self.target_list[i] = None

    def publish_info(self):
        msg = Float32MultiArray()

        # Add the timestamp
        msg.layout.dim.append(MultiArrayDimension(label="timestamp", size=1, stride=1))
        msg.data.append(self.current_timestamp)

        # Add the reward
        msg.layout.dim.append(MultiArrayDimension(label="reward", size=1, stride=1))
        msg.data.append(self.current_reward)

        # Publish the message
        self.info_publisher.publish(msg)


class TARGET:
    def __init__(self, x: float, y: float, z: float, name: str):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def get_distance(self, state):
        # Calculate the Euclidean distance between the target and the state,
        # note that we only consider the x and y coordinates.
        return math.sqrt((self.x - state.pose.position.x) ** 2 + (self.y - state.pose.position.y) ** 2)
