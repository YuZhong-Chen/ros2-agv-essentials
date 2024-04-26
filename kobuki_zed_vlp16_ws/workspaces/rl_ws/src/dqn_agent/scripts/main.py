#!/usr/bin/env python3

import rclpy

import os
import cv2
import time
import datetime
import math
import torch
import threading
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dqn_agent.rl_env import RL_ENV, OBSERVATION_SUBSCRIBER, ENV_INFO_SUBSCRIBER
from dqn_agent.wrapper import RESIZE_OBSERVATION, GRAY_SCALE_OBSERVATION, FRAME_STACKING, BLUR_OBSERVATION
from dqn_agent.agent import AGENT
from dqn_agent.logger import LOGGER

#############################################################################################
EPISODES = 10000
SAVE_INTERVAL = 50

PROJECT = "rl_dqn"
PROJECT_NAME = PROJECT + "-" + datetime.datetime.now().strftime("%m-%d-%H-%M")

IS_LOAD_MODEL = False
LOAD_MODEL_PATH = "rl_dqn-04-18-01-20/models/episode_4"

USE_LOGGER = True
USE_WANDB = False
#############################################################################################

checkpoint_dir = Path("/home/ros2-agv-essentials/rl_ws/checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
project_dir = checkpoint_dir / PROJECT_NAME
project_dir.mkdir(exist_ok=True)

agent = AGENT(is_load=IS_LOAD_MODEL, load_path=(checkpoint_dir / LOAD_MODEL_PATH))
logger = LOGGER(project=PROJECT, project_name=PROJECT_NAME, config=agent.config, project_dir=project_dir, enable=USE_LOGGER, use_wandb=USE_WANDB)

RESIZE_OBSERVATION_ = RESIZE_OBSERVATION(shape=(320, 320))
GRAY_SCALE_OBSERVATION_ = GRAY_SCALE_OBSERVATION()
FRAME_STACKING_ = FRAME_STACKING(stack_size=4)
BLUR_OBSERVATION_ = BLUR_OBSERVATION(kernel_size=5)


def ProcessObservation(observation):
    # observation = observation.squeeze()
    observation = BLUR_OBSERVATION_.forward(observation)
    observation = RESIZE_OBSERVATION_.forward(observation)
    observation = GRAY_SCALE_OBSERVATION_.forward(observation)
    observation = FRAME_STACKING_.forward(observation)
    return observation


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

    for episode in tqdm(range(agent.current_episode + 1, EPISODES + 1)):
        observation = env.reset()
        observation = ProcessObservation(observation)

        episode_step = 0
        episode_reward = 0
        loss_list = []
        td_error_list = []
        td_estimation_list = []

        while True:
            action = agent.Act(observation)

            next_observation, reward, done = env.step(action)
            next_observation = ProcessObservation(next_observation)

            agent.AddToReplayBuffer(observation, action, reward, next_observation)

            loss, td_error, td_estimation = agent.Train()

            episode_reward += reward
            loss_list.append(loss)
            td_error_list.append(td_error)
            td_estimation_list.append(td_estimation)

            observation = next_observation

            episode_step += 1
            if episode_step > 70:
                break

        agent.current_episode = episode
        average_loss = np.mean(loss_list)
        average_td_error = np.mean(td_error_list)
        average_td_estimation = np.mean(td_estimation_list)

        logger.Log(episode, average_loss, average_td_error, average_td_estimation, episode_reward)
        env.get_logger().info(f"Episode: {episode}, Reward: {round(episode_reward, 3)}, Loss: {round(average_loss, 3)}, TD Error: {round(average_td_error, 3)}, TD Estimation: {round(average_td_estimation, 3)}")


if __name__ == "__main__":
    main()
