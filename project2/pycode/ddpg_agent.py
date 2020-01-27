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

from pycode.model import QNetwork, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 4*64       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-4     # L2 weight decay
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
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
        self.noise = OUNoise((1, action_size), seed)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
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
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # if random.random() > noise:
        #     # print('if ********************************')
        #     # print(action_values.cpu().data.numpy()[0])
        #     # print('end if ********************************')
        #     return action_values.cpu().data.numpy()[0]
        # else:
        #     # print('else ##############################')
        #     # print(np.clip(np.random.normal(loc=action_values.cpu().data.numpy()[0], scale=noise), -1, 1))
        #     # print('end else ##############################')
        #     return np.clip(np.random.normal(loc=action_values.cpu().data.numpy()[0], scale=noise), -1, 1)
        # Epsilon-greedy action selection
        actions = action_values.cpu().data.numpy()[0]
        if noise:
            actions += self.noise.sample()[0]
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        actions_next = self.qnetwork_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        #    next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.critic_local(states, actions)
        # if Q_expected.shape != Q_targets.shape:
        # print('qexpected', Q_expected.shape)
        # print('qtargets', Q_targets.shape)
        # print('rewards', rewards.shape)
        # print('dones', dones.shape)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # print(Q_targets.shape, Q_expected.shape)

        # Compute loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # loss = F.mse_loss(Q_expected, Q_targets)

        actions_pred = self.qnetwork_local(states)
        qnetwork_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.qnetwork_optimizer.zero_grad()
        qnetwork_loss.backward()
        self.qnetwork_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

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
        torch.save(self.qnetwork_local.state_dict(), "model/weights_local.torch")
        torch.save(self.qnetwork_target.state_dict(), "model/weights_target.torch")
        torch.save(self.critic_local.state_dict(), "model/critic_local.torch")
        torch.save(self.critic_target.state_dict(), "model/critic_target.torch")

    def load_model(self):
        self.qnetwork_local.load_state_dict(torch.load("model/weights_local.torch", map_location=device))
        self.qnetwork_target.load_state_dict(torch.load("model/weights_target.torch", map_location=device))
        self.critic_local.load_state_dict(torch.load("model/critic_local.torch", map_location=device))
        self.critic_target.load_state_dict(torch.load("model/critic_local.torch", map_location=device))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

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

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.25, sigma_min=0.01, sigma_decay=0.98):
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
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
