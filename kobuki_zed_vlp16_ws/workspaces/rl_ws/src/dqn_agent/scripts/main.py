#!/usr/bin/env python3

import rclpy
from dqn_agent.rl_env import RL_ENV

import time
import numpy as np


def main(args=None):
    rclpy.init(args=args)

    env = RL_ENV()

    env.reset()

    while True:
        observation, reward, done = env.step(np.random.randint(0, env.action_space))
        env.get_logger().info(f"Observation: {observation.shape}, Reward: {reward}, Done: {done}")


if __name__ == "__main__":
    main()
