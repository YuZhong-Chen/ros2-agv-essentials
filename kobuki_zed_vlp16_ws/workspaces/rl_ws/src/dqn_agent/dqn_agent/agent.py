import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

from dqn_agent.dqn import DQN
from dqn_agent.per import PRIORITIZED_EXPERIENCE_REPLAY


class AGENT:
    def __init__(self, is_load=False, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = {
            "batch_size": 128,
            "learning_rate": 0.0004,
            "gamma": 0.85,
            "replay_buffer_size": 5000,
            "warmup_steps": 500,
            "tau": 0.001,
            "optimizer": "AdamW",
            "loss": "MSE",
        }

        self.network = None
        self.replay_buffer = None
        self.optimizer = None
        self.loss = None
        self.action_space_len = 6

        self.last_action = None
        self.current_step = 0
        self.current_episode = 0

        if is_load:
            self.LoadModel(load_path)
        else:
            self.Init()

    def Init(self):
        print("Init Agent ...")

        self.network = DQN(tau=self.config["tau"])
        self.replay_buffer = PRIORITIZED_EXPERIENCE_REPLAY(capacity=self.config["replay_buffer_size"])

        if self.config["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.network.learning_network.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "AdamW":
            self.optimizer = optim.AdamW(self.network.learning_network.parameters(), lr=self.config["learning_rate"], amsgrad=True)
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(self.network.learning_network.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(self.network.learning_network.parameters(), lr=self.config["learning_rate"])

        if self.config["loss"] == "MSE":
            self.loss = nn.MSELoss(reduction="mean")
        elif self.config["loss"] == "L1Loss":
            self.loss = nn.L1Loss()
        elif self.config["loss"] == "SmoothL1Loss":
            self.loss = nn.SmoothL1Loss()

        print("Model config: ", self.config)

    def Act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.int8).unsqueeze(0).to(self.device)
            action = torch.argmax(self.network.learning_network(state)).item()
        return action

    def AddToReplayBuffer(self, state, action, reward, next_state):
        self.replay_buffer.Add(state, action, reward, next_state)

    def Train(self):
        self.current_step += 1
        
        if self.current_step < self.config["warmup_steps"]:
            return 0, 0, 0

        # Sample batch data from replay buffer.
        state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.Sample(self.config["batch_size"])

        # Transfer the data type of action (int8) to (int64) for gather() function.
        action_batch = action_batch.to(torch.int64)

        # Calculate TD target.
        with torch.no_grad():
            max_action = torch.argmax(self.network.learning_network(next_state_batch), dim=1)
            next_q = self.network.target_network(next_state_batch).gather(1, max_action.unsqueeze(-1))
            td_target = reward_batch + self.config["gamma"] * next_q

        # Calculate TD estimation.
        td_estimation = self.network.learning_network(state_batch).gather(1, action_batch)

        # Compute loss
        loss = self.loss(td_estimation, td_target)

        # Update network.
        # Clip the gradient to prevent the gradient explosion.
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.learning_network.parameters(), 50.0)
        self.optimizer.step()

        # Update target network. (Soft update)
        self.network.UpdateTargetNetwork()

        # Update the priority of the replay buffer.
        with torch.no_grad():
            td_error = torch.abs(td_target - td_estimation).cpu()
            self.replay_buffer.UpdatePriority(td_error)

        return loss.item(), td_error.mean().item(), td_estimation.mean().item()

    def SaveModel(self, path):
        print("Model saved at", path)

        # Save model parameters and config
        learning_network_path = path / "learning_network.pth"
        target_network_path = path / "target_network.pth"
        config_path = path / "config.pth"

        torch.save({"model": self.network.learning_network.state_dict()}, learning_network_path)
        torch.save({"model": self.network.target_network.state_dict()}, target_network_path)
        torch.save({"config": self.config}, config_path)

        print(self.config)

    def LoadModel(self, path):
        print("Load model from", path)

        config_path = path / "config.pth"
        checkpoint = torch.load(config_path)
        self.config = checkpoint["config"]

        # If you want to modify the model's configuration,
        # add the configuration to the config here directly.
        # Ex: self.config["batch_size"] = 128

        self.Init()

        learning_network_path = path / "learning_network.pth"
        checkpoint = torch.load(learning_network_path)
        self.network.learning_network.load_state_dict(checkpoint["model"])

        target_network_path = path / "target_network.pth"
        checkpoint = torch.load(target_network_path)
        self.network.target_network.load_state_dict(checkpoint["model"])

        print("Model loaded successfully.")
