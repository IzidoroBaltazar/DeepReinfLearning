"""
originally taken from
https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py
Modified for project2 based on
https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal
"""
import copy
import numpy as np
import random
from collections import namedtuple, deque

# from pycode.model import QNetwork, Critic
from pycode.model import QNetwork, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 10**4     # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_agents=1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.num_agents = num_agents

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        # weight_decay=WEIGHT_DECAY
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process for each agent
        self.noise = OUNoise((self.num_agents, action_size), seed)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # for agent in range(self.num_agents):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            noise (float): noise level
        """
        state = torch.from_numpy(state).float().to(device)
        action_values = np.zeros((self.num_agents, self.action_size))
        self.qnetwork_local.eval()
        with torch.no_grad():
            for agent in range(self.num_agents):
                action_values[agent, :] = self.qnetwork_local(state[agent, :]).cpu().data.numpy()
        self.qnetwork_local.train()

        actions = action_values
        if noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            (all of the experiences are actions of both agents stacked)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        for agent in range(self.num_agents):
            opponent = (agent + 1) % 2
            # Get max predicted Q values (for next states) from target model
            actions_next_agent = self.qnetwork_target(next_states[agent])
            Q_targets_next = self.critic_target(next_states[agent], actions_next_agent)
            actions_next_opponent = self.qnetwork_target(next_states[opponent])
            Q_targets_opponent = torch.flatten(
                self.critic_target(next_states[opponent], actions_next_opponent))
            Q_targets_agent = torch.flatten(
                self.critic_target(next_states[agent], actions_next_agent))
            # Compute Q targets for current states
            # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Q_targets_o = rewards[opponent] + (gamma * Q_targets_opponent * (1 - dones[opponent]))
            Q_targets_a = rewards[agent] + (gamma * Q_targets_agent * (1 - dones[agent]))
            Q_targets = Q_targets_a + rewards[opponent]
            # Q_targets = Q_targets_a + rewards[opponent]
            # import pdb; pdb.set_trace()

            # Get expected Q values from local model
            Q_expected_a = torch.flatten(self.critic_local(states[agent], actions[agent]))
            # Q_expected_o = torch.flatten(self.critic_local(states[opponent], actions[opponent]))
            # Q_expected = Q_expected_a + Q_expected_o
            Q_expected = Q_expected_a
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Compute loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actions_pred_a = self.qnetwork_local(states[agent])
            # actions_pred_o = self.qnetwork_local(states[opponent])
            qnetwork_loss_a = -self.critic_local(states[agent], actions_pred_a).mean()
            # qnetwork_loss_o = -self.critic_local(states[opponent], actions_pred_o).mean()
            # qnetwork_loss = qnetwork_loss_a - rewards[opponent]
            # qnetwork_loss = qnetwork_loss_a.mean()
            # qnetwork_loss = qnetwork_loss_a + qnetwork_loss_o
            # import pdb; pdb.set_trace()
            qnetwork_loss = qnetwork_loss_a
            # Minimize the loss
            self.qnetwork_optimizer.zero_grad()
            qnetwork_loss.backward()
            # qnetwork_loss = qnetwork_loss_o
            # self.qnetwork_optimizer.zero_grad()
            # qnetwork_loss.backward()
            self.qnetwork_optimizer.step()

            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self):
        torch.save(self.qnetwork_local.state_dict(),
                   "model/weights_local.torch")
        torch.save(self.qnetwork_target.state_dict(),
                   "model/weights_target.torch")
        torch.save(self.critic_local.state_dict(), "model/critic_local.torch")
        torch.save(self.critic_target.state_dict(),
                   "model/critic_target.torch")

    def load_model(self):
        self.qnetwork_local.load_state_dict(torch.load(
            "model/weights_local.torch", map_location=device))
        self.qnetwork_target.load_state_dict(torch.load(
            "model/weights_target.torch", map_location=device))
        self.critic_local.load_state_dict(torch.load(
            "model/critic_local.torch", map_location=device))
        self.critic_target.load_state_dict(torch.load(
            "model/critic_local.torch", map_location=device))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.average_reward = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        # if (np.absolute(e.reward).sum()) > 0.:
        self.memory.append(e)
        # self.average_reward = (e.reward.sum() + self.average_reward) / 2
        # self.memory.append(e)
        # import pdb; pdb.set_trace()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None], axis=1)).float().to(device)
        actions = torch.from_numpy(
            np.stack([e.action for e in experiences if e is not None], axis=1)).float().to(device)
        rewards = torch.from_numpy(
            np.stack([e.reward for e in experiences if e is not None], axis=1)).float().to(device)
        next_states = torch.from_numpy(np.stack(
            [e.next_state for e in experiences if e is not None], axis=1)).float().to(device)
        dones = torch.from_numpy(np.stack(
            [e.done for e in experiences if e is not None], axis=1).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.15, sigma_min=0.025, sigma_decay=.999):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Sigma reduction"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
