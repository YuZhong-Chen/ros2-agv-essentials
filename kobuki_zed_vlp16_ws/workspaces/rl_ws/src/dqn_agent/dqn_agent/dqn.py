import torch
import math
import torch.nn as nn


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__(in_features, out_features)

        self.sigma = 0.5

        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(self.sigma / math.sqrt(self.in_features)))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(self.sigma / math.sqrt(self.in_features)))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight.data.uniform_(-mu_range, mu_range)
        self.bias.data.uniform_(-mu_range, mu_range)

    def sample_noise(self):
        self.weight_epsilon = torch.randn(self.weight_epsilon.size(), device=self.device)
        self.bias_epsilon = torch.randn(self.bias_epsilon.size(), device=self.device)

    def forward(self, x):
        self.sample_noise()
        return nn.functional.linear(x, self.weight + self.weight_sigma * self.weight_epsilon, self.bias + self.bias_sigma * self.bias_epsilon)


class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()

        # State shape: 4x320x320 (4 frames of 320x320 pixels)
        # Action space: 6

        self.feature_map = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=10, stride=3, padding=0),  # 4x320x320 -> 16x104x104
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=10, stride=3, padding=0),  # 16x104x104 -> 32x32x32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),  # 32x32x32 -> 64x14x14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=0),  # 64x14x14 -> 64x5x5
            nn.LeakyReLU(),
            nn.Flatten(),  # 64x5x5 -> 1600
        )

        self.advantage = nn.Sequential(
            NoisyLinear(1600, 512),
            nn.LeakyReLU(),
            NoisyLinear(512, 6),
        )

        self.value = nn.Sequential(
            NoisyLinear(1600, 512),
            nn.LeakyReLU(),
            NoisyLinear(512, 1),
        )

    def forward(self, x):
        # Transform the range of x from [0, 255] to [0, 1]
        x = x / 255.0

        x = self.feature_map(x)

        advantage = self.advantage(x)
        value = self.value(x)

        # Dueling DQN -> Q(s, a) = V(s) + A(s, a)
        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_value


class DQN:
    def __init__(self, tau=0.001):
        # Network
        self.learning_network = None
        self.target_network = None

        self.tau = 0.001
        self.tau_minus = 1.0 - self.tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Init(tau)

    def Init(self, tau=0.001):
        self.learning_network = NETWORK().to(self.device)
        self.target_network = NETWORK().to(self.device)

        self.tau = tau
        self.tau_minus = 1.0 - self.tau

        # Init target_network's parameters
        self.target_network.load_state_dict(self.learning_network.state_dict())

        # Frozen target_network's parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

    def UpdateTargetNetwork(self):
        # Use soft update
        for target_parameter, learning_parameter in zip(self.target_network.parameters(), self.learning_network.parameters()):
            target_parameter.data.copy_(self.tau * learning_parameter.data + self.tau_minus * target_parameter.data)
