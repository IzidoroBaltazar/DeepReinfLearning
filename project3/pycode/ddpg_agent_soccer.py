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
from pycode.model_soccer import QNetwork, Critic

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


class SoccerAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, seed, num_agents,
                       g_brain_name, g_state_size, g_action_size,
                       s_brain_name, s_state_size, s_action_size,
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # print('state_size', state_size)
        self.g_brain_name = g_brain_name
        self.g_state_size = g_state_size
        self.g_action_size = g_action_size

        self.seed = random.seed(seed)
        self.num_agents = num_agents

        self.s_brain_name = s_brain_name
        self.s_state_size = s_state_size
        self.s_action_size = s_action_size
        # self.kind = kind

        # Q-Network
        self.g_qnetwork_local = QNetwork(g_state_size, g_action_size, seed).to(device)
        self.g_qnetwork_target = QNetwork(g_state_size, g_action_size, seed).to(device)
        self.g_qnetwork_optimizer = optim.Adam(self.g_qnetwork_local.parameters(), lr=LR)

        # Critic Network (w/ Target Network)
        self.g_critic_local = Critic(g_state_size, g_action_size, seed).to(device)
        self.g_critic_target = Critic(g_state_size, g_action_size, seed).to(device)
        # weight_decay=WEIGHT_DECAY
        self.g_critic_optimizer = optim.Adam(self.g_critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Q-Network
        self.s_qnetwork_local = QNetwork(s_state_size, s_action_size, seed).to(device)
        self.s_qnetwork_target = QNetwork(s_state_size, s_action_size, seed).to(device)
        self.s_qnetwork_optimizer = optim.Adam(self.s_qnetwork_local.parameters(), lr=LR)

        # Critic Network (w/ Target Network)
        self.s_critic_local = Critic(s_state_size, s_action_size, seed).to(device)
        self.s_critic_target = Critic(s_state_size, s_action_size, seed).to(device)
        # weight_decay=WEIGHT_DECAY
        self.s_critic_optimizer = optim.Adam(self.s_critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process for each agent
        # self.noise = OUNoise((self.num_agents, action_size), seed)
        # Replay memory
        self.memory = ReplayBuffer(g_action_size + s_action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self,
             g_state, g_action, g_reward, g_next_state, done,
             s_state, s_action, s_reward, s_next_state
             ):
        # Save experience in replay memory
        # for agent in range(self.num_agents):
        # g_action = []
        # import pdb; pdb.set_trace()
        a, b = to_vector(g_action[0], self.g_action_size), to_vector(g_action[1], self.g_action_size)
        g_action = [a, b]
        a, b = to_vector(s_action[0], self.s_action_size), to_vector(s_action[1], self.s_action_size)
        s_action = [a, b]
        self.memory.add(
            g_state, g_action, g_reward, g_next_state, done,
            s_state, s_action, s_reward, s_next_state
        )

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, g_state, s_state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            noise (float): noise level
        """
        g_state = torch.from_numpy(g_state).float().to(device)
        g_action_values = np.zeros((self.num_agents, self.g_action_size))
        self.g_qnetwork_local.eval()
        with torch.no_grad():
            for agent in range(self.num_agents):
                g_action_values[agent, :] = self.g_qnetwork_local(g_state[agent, :]).cpu().data.numpy()
        self.g_qnetwork_local.train()

        g_actions = g_action_values

        s_state = torch.from_numpy(s_state).float().to(device)
        s_action_values = np.zeros((self.num_agents, self.s_action_size))
        self.s_qnetwork_local.eval()
        with torch.no_grad():
            for agent in range(self.num_agents):
                s_action_values[agent, :] = self.s_qnetwork_local(s_state[agent, :]).cpu().data.numpy()
        self.s_qnetwork_local.train()

        s_actions = s_action_values

        # print('action_values', action_values)
        # print('type actions: ', type(actions))
        if random.random() > eps:
            return {
                    self.g_brain_name: [np.argmax(action) for action in g_actions],
                    self.s_brain_name: [np.argmax(action) for action in g_actions],
                    }
        else:
            return {
                self.g_brain_name: [random.choice(np.arange(self.g_action_size)) for _ in range(self.num_agents)],
                self.s_brain_name: [random.choice(np.arange(self.s_action_size)) for _ in range(self.num_agents)],
            }

    def reset(self):
        pass
        # self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        g_states, g_actions, g_rewards, g_next_states, g_dones, s_states, s_actions, s_rewards, s_next_states, s_dones = experiences

        for agent in range(self.num_agents):
            opponent = (agent + 1) % 2
            # Get max predicted Q values (for next states) from target model
            g_actions_next_agent = self.g_qnetwork_target(g_next_states[agent])
            s_actions_next_agent = self.s_qnetwork_target(s_next_states[agent])
            # g_actions_next_agent = to_vector(g_actions_next_agent, self.g_action_size)
            # s_actions_next_agent = to_vector(s_actions_next_agent, self.s_action_size)
            # print(actions_next_agent, next_states[agent])
            g_Q_targets_next = self.g_critic_target(g_next_states[agent], g_actions_next_agent)
            s_Q_targets_next = self.s_critic_target(s_next_states[agent], s_actions_next_agent)
            g_actions_next_opponent = self.g_qnetwork_target(g_next_states[opponent])
            s_actions_next_opponent = self.s_qnetwork_target(s_next_states[opponent])
            g_Q_targets_opponent = torch.flatten(
                self.g_critic_target(g_next_states[opponent], g_actions_next_opponent))
            s_Q_targets_opponent = torch.flatten(
                self.s_critic_target(s_next_states[opponent], s_actions_next_opponent))
            g_Q_targets_agent = torch.flatten(
                self.g_critic_target(g_next_states[agent], g_actions_next_agent))
            s_Q_targets_agent = torch.flatten(
                self.s_critic_target(s_next_states[agent], s_actions_next_agent))
            # Compute Q targets for current states
            # print(Q_targets_opponent.shape)
            # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Q_targets_o = rewards[opponent] + (gamma * Q_targets_opponent * (1 - dones[opponent]))
            g_Q_targets_a = g_rewards[agent] + (gamma * g_Q_targets_agent * (1 - g_dones[agent]))
            g_Q_targets = g_Q_targets_a + s_rewards[agent] - s_rewards[opponent] - g_rewards[opponent]
            # Q_targets = Q_targets_a + rewards[opponent]
            s_Q_targets_a = s_rewards[agent] + (gamma * s_Q_targets_agent * (1 - s_dones[agent]))
            s_Q_targets = s_Q_targets_a + s_rewards[agent] - s_rewards[opponent] - g_rewards[opponent]

            # Get expected Q values from local model
            # import pdb; pdb.set_trace()
            # g_actions[agent] = to_vector(g_actions[agent], self.g_state_size)
            # s_actions[agent] = to_vector(s_actions[agent], self.s_state_size)
            g_Q_expected_a = torch.flatten(self.g_critic_local(g_states[agent], g_actions[agent]))
            s_Q_expected_a = torch.flatten(self.s_critic_local(s_states[agent], s_actions[agent]))
            # Q_expected_o = torch.flatten(self.critic_local(states[opponent], actions[opponent]))
            # Q_expected = Q_expected_a + Q_expected_o
            g_Q_expected = g_Q_expected_a
            g_critic_loss = F.mse_loss(g_Q_expected, g_Q_targets)

            s_Q_expected = s_Q_expected_a
            s_critic_loss = F.mse_loss(s_Q_expected, s_Q_targets)

            # Compute loss
            self.g_critic_optimizer.zero_grad()
            g_critic_loss.backward()
            self.g_critic_optimizer.step()

            self.s_critic_optimizer.zero_grad()
            s_critic_loss.backward()
            self.s_critic_optimizer.step()

            g_actions_pred_a = self.g_qnetwork_local(g_states[agent])
            # actions_pred_o = self.qnetwork_local(states[opponent])
            g_qnetwork_loss_a = -self.g_critic_local(g_states[agent], g_actions_pred_a).mean()

            s_actions_pred_a = self.s_qnetwork_local(s_states[agent])
            # actions_pred_o = self.qnetwork_local(states[opponent])
            s_qnetwork_loss_a = -self.s_critic_local(s_states[agent], s_actions_pred_a).mean()
            # qnetwork_loss_o = -self.critic_local(states[opponent], actions_pred_o).mean()
            # qnetwork_loss = qnetwork_loss_a - rewards[opponent]
            # qnetwork_loss = qnetwork_loss_a.mean()
            # qnetwork_loss = qnetwork_loss_a + qnetwork_loss_o
            # import pdb; pdb.set_trace()
            g_qnetwork_loss = g_qnetwork_loss_a
            # Minimize the loss
            self.g_qnetwork_optimizer.zero_grad()
            g_qnetwork_loss.backward()
            # qnetwork_loss = qnetwork_loss_o
            # self.qnetwork_optimizer.zero_grad()
            # qnetwork_loss.backward()
            self.g_qnetwork_optimizer.step()

            s_qnetwork_loss = s_qnetwork_loss_a
            # Minimize the loss
            self.s_qnetwork_optimizer.zero_grad()
            s_qnetwork_loss.backward()
            self.s_qnetwork_optimizer.step()

            self.soft_update(self.g_critic_local, self.g_critic_target, TAU)
            self.soft_update(self.g_qnetwork_local, self.g_qnetwork_target, TAU)

            self.soft_update(self.s_critic_local, self.s_critic_target, TAU)
            self.soft_update(self.s_qnetwork_local, self.s_qnetwork_target, TAU)

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
        torch.save(self.g_qnetwork_local.state_dict(),
                   "model/soccer_g_weights_local.torch")
        torch.save(self.g_qnetwork_target.state_dict(),
                   "model/soccer_g_weights_target.torch")
        torch.save(self.g_critic_local.state_dict(),
                   "model/soccer_g_critic_local.torch")
        torch.save(self.g_critic_target.state_dict(),
                   "model/soccer_g_critic_target.torch")

        torch.save(self.s_qnetwork_local.state_dict(),
                   "model/soccer_s_weights_local.torch")
        torch.save(self.s_qnetwork_target.state_dict(),
                   "model/soccer_s_weights_target.torch")
        torch.save(self.s_critic_local.state_dict(),
                   "model/soccer_s_critic_local.torch")
        torch.save(self.s_critic_target.state_dict(),
                   "model/soccer_s_critic_target.torch")

    def load_model(self):
        self.g_qnetwork_local.load_state_dict(torch.load(
            "model/soccer_g_weights_local.torch", map_location=device))
        self.g_qnetwork_target.load_state_dict(torch.load(
            "model/soccer_g_weights_target.torch", map_location=device))
        self.g_critic_local.load_state_dict(torch.load(
            "model/soccer_g_critic_local.torch", map_location=device))
        self.g_critic_target.load_state_dict(torch.load(
            "model/soccer_g_critic_local.torch", map_location=device))

        self.s_qnetwork_local.load_state_dict(torch.load(
            "model/soccer_s_weights_local.torch", map_location=device))
        self.s_qnetwork_target.load_state_dict(torch.load(
            "model/soccer_s_weights_target.torch", map_location=device))
        self.s_critic_local.load_state_dict(torch.load(
            "model/soccer_s_critic_local.torch", map_location=device))
        self.s_critic_target.load_state_dict(torch.load(
            "model/soccer_s_critic_local.torch", map_location=device))


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
                                     "g_state", "g_action", "g_reward", "g_next_state", "done",
                                     "s_state", "s_action", "s_reward", "s_next_state"])
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.average_reward = 0

    def add(self,
            g_state, g_action, g_reward, g_next_state, done,
            s_state, s_action, s_reward, s_next_state,
            ):
        """Add a new experience to memory."""
        e = self.experience(
            g_state, g_action, g_reward, g_next_state, done,
            s_state, s_action, s_reward, s_next_state,
        )
        # if (np.absolute(e.reward).sum()) > 0.:
        self.memory.append(e)
        # self.average_reward = (e.reward.sum() + self.average_reward) / 2
        # self.memory.append(e)
        # import pdb; pdb.set_trace()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        g_states = torch.from_numpy(
            np.stack([e.g_state for e in experiences if e is not None], axis=1)).float().to(device)
        g_actions = torch.from_numpy(
            np.stack([e.g_action for e in experiences if e is not None], axis=1).astype(np.uint8)).float().to(device)
        g_rewards = torch.from_numpy(
            np.stack([e.g_reward for e in experiences if e is not None], axis=1)).float().to(device)
        g_next_states = torch.from_numpy(np.stack(
            [e.g_next_state for e in experiences if e is not None], axis=1)).float().to(device)
        dones = torch.from_numpy(np.stack(
            [e.done for e in experiences if e is not None], axis=1).astype(np.uint8)).float().to(device)

        s_states = torch.from_numpy(
            np.stack([e.s_state for e in experiences if e is not None], axis=1)).float().to(device)
        s_actions = torch.from_numpy(
            np.stack([e.s_action for e in experiences if e is not None], axis=1).astype(np.uint8)).float().to(device)
        s_rewards = torch.from_numpy(
            np.stack([e.s_reward for e in experiences if e is not None], axis=1)).float().to(device)
        s_next_states = torch.from_numpy(np.stack(
            [e.s_next_state for e in experiences if e is not None], axis=1)).float().to(device)

        return (
            g_states, g_actions, g_rewards, g_next_states, dones,
            s_states, s_actions, s_rewards, s_next_states, dones
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def to_vector(state, state_size):
    # state = state.cpu().data.numpy()
    # import pdb; pdb.set_trace()
    ret = np.zeros([state_size])
    # import pdb; pdb.set_trace()
    ret[int(state)] = 1
    return ret

