#!/usr/bin/env python3

import rclpy
from gazebo_rl_env.gazebo_rl_env import GAZEBO_RL_ENV_NODE, RESET_SERVICE, STEP_SERVICE

import time
import threading


def main(args=None):
    rclpy.init(args=args)

    reset_service_node = RESET_SERVICE()
    step_service_node = STEP_SERVICE()

    # Create a MultiThreadedExecutor
    # This is necessary for running the services in parallel,
    # otherwise the services will be blocked by each other.
    # Reference: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html
    # TODO: This is a temporary solution, and it may not be the best solution,
    # since the services use busy waiting, it may consume a lot of CPU resources.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(reset_service_node)
    executor.add_node(step_service_node)
    executor_thread = threading.Thread(target=executor.spin)
    executor_thread.start()

    gazebo_rl_env = GAZEBO_RL_ENV_NODE()

    while True:
        # TODO: This is a temporary solution, and it may not be the best solution,
        time.sleep(0.05)

        if reset_service_node.is_reset:
            gazebo_rl_env.reset()
            reset_service_node.is_reset = False
        if step_service_node.is_step:
            gazebo_rl_env.step()
            step_service_node.is_step = False


if __name__ == "__main__":
    main()
