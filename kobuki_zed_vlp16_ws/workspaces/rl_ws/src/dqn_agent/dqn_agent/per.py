import torch
import math
import numpy as np


class PRIORITIZED_EXPERIENCE_REPLAY:
    def __init__(self, alpha=0.6, epsilon=0.01, capacity=50000, state_shape=(4, 320, 320)):
        self.alpha = alpha
        self.epsilon = epsilon
        self.capacity = capacity
        self.current_size = 0
        self.sample_index = None
        self.replace_index = 0

        # Store the transitions in GPU memory if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the replay buffer
        # Note that the state is stored as int8 to save memory.
        self.state = torch.zeros((capacity, *state_shape), dtype=torch.int8).to(self.device)
        self.action = torch.zeros((capacity, 1), dtype=torch.int8).to(self.device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.int8).to(self.device)
        self.next_state = torch.zeros((capacity, *state_shape), dtype=torch.int8).to(self.device)
        self.priority = torch.ones(capacity, dtype=torch.float32)

    def Sample(self, batch_size=32):
        # Here I use the torch.multinomial function to sample the index of the transitions.
        # Normally, it should be implemented with a SumTree data structure.
        # In order to make the code simple, and speed up the training process, I use the torch.multinomial function.
        self.sample_index = torch.multinomial(self.priority[:self.current_size], batch_size, replacement=True)
        return self.state[self.sample_index], self.action[self.sample_index], self.reward[self.sample_index], self.next_state[self.sample_index]

    def Add(self, state, action, reward, next_state):
        self.state[self.replace_index] = torch.tensor(np.array(state), dtype=torch.int8)
        self.action[self.replace_index] = action
        self.reward[self.replace_index] = reward
        self.next_state[self.replace_index] = torch.tensor(np.array(next_state), dtype=torch.int8)
        self.priority[self.replace_index] = self.priority.max()

        self.replace_index = (self.replace_index + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def UpdatePriority(self, priority):
        for i, index in enumerate(self.sample_index):
            self.priority[index] = (priority[i] + self.epsilon) ** self.alpha