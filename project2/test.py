from unityagents import UnityEnvironment
from tqdm import trange
import numpy as np
import torch
import sys
# agent code from
from pycode import QNetwork, Agent, ReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if sys.platform == "darwin":
    env = UnityEnvironment(file_name="./Reacher.app")
else:
    env = UnityEnvironment(file_name="Reacher_Linux_NoVis/Reacher.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print(brain_name)
print(brain)

state_size = brain.vector_observation_space_size
print("State size: ", state_size)

# reset the environment
# env_info = env.reset(train_mode=True)[brain_name]
env_info = env.reset()[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

env_info.vector_observations.shape

agent = Agent(state_size=state_size, action_size=action_size, seed=43)
agent.load_model()
num_agents = len(env_info.agents)

env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
env_info
j = 0
mx = 100

total_scores = []
improvement = False

with open('data-test.csv', 'w') as f:
    f.write("Index,Score,Rolling avg score\n")

stop = 0
max_value = -10  # start with negative score to prevent early training stop
t = trange(1, mx+1)

for i in t:
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations             # get the current state
    # states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    agent.reset()

    while True:
        # for state in states:
        actions = agent.act(states, noise=False)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states

        if np.any(dones):
            break

    total_scores.append(np.mean(scores))
    avg_score = np.mean(total_scores[i-min(i,100):i+1])
    with open('data-test.csv', 'a+') as f:
        f.write("{},{:.3f},{:.3f}\n".format(i, scores[0], avg_score))

    env.reset(train_mode=True)
