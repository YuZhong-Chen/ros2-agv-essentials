#!/usr/bin/env python3

import rclpy
from dqn_agent.rl_env import RL_ENV, OBSERVATION_SUBSCRIBER, ENV_INFO_SUBSCRIBER

import time
import threading
import numpy as np


def main(args=None):
    rclpy.init(args=args)
    
    observation_subscriber_node = OBSERVATION_SUBSCRIBER()
    env_info_subscriber_node = ENV_INFO_SUBSCRIBER()
    
    # Create a MultiThreadedExecutor
    # Reference: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html
    # TODO: This is a temporary solution, and it may not be the best solution,
    # since the main thread use busy waiting, it may consume a lot of CPU resources.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(observation_subscriber_node)
    executor.add_node(env_info_subscriber_node)
    executer_thread = threading.Thread(target=executor.spin)
    executer_thread.start()

    env = RL_ENV()

    env.reset()

    while True:
        observation, reward, done = env.step(np.random.randint(0, env.action_space))
        env.get_logger().info(f"Observation: {observation.shape}, Reward: {reward}, Done: {done}")


if __name__ == "__main__":
    main()
