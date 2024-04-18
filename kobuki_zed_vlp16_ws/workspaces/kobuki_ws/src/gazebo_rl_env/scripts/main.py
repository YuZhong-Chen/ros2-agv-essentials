#!/usr/bin/env python3

import rclpy
from gazebo_rl_env.gazebo_rl_env import GAZEBO_RL_ENV_NODE

import time

def main(args=None):
    rclpy.init(args=args)

    gazebo_rl_env = GAZEBO_RL_ENV_NODE()

    gazebo_rl_env.reset()
    
    while True:
        gazebo_rl_env.step()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
