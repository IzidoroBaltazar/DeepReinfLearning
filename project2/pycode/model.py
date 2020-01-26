"""
originally taken from
https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc2_units, action_size)
        # self.softmax = nn.Softmax(dim=1)
        lim = 1 / np.sqrt(fc1_units)
        self.fc1.weight.data.uniform_(-lim, lim)
        lim = 1 / np.sqrt(fc2_units)
        self.fc2.weight.data.uniform_(-lim, lim)
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
        # return self.softmax(x)
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc2_units, 1)
        lim = 1 / np.sqrt(fc1_units)
        self.fc1.weight.data.uniform_(-lim, lim)
        lim = 1 / np.sqrt(fc2_units)
        self.fc2.weight.data.uniform_(-lim, lim)
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        # print(x.dtype)
        # print(action.dtype)
        # print(action.shape)
        #f print(action)
        # action = torch.FloatTensor(action)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
        # return self.softmax(x)
        return self.fc4(x)
