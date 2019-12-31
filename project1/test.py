from unityagents import UnityEnvironment
from tqdm import tqdm
import numpy as np
import torch
from torch import FloatTensor, LongTensor, cuda
import sys
# agent code from
from pycode import QNetwork, Agent, ReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if sys.platform == "darwin":
    env = UnityEnvironment(file_name="./Banana.app")
else:
    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    # env = UnityEnvironment(file_name="./Banana_Linux")

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

env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
env_info
j = 0
eps = 0.
mx = 100

total_scores = []
improvement = False

with open('data-test.csv', 'w') as f:
    f.write("Index,Score,Exploration,Rolling avg score\n")

stop = 0
max_value = -10  # start with negative score to prevent early training stop

for i in tqdm(range(mx)):
    states, actions, rewards, next_states, dones = [[], [], [], [], []]
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        agent.step(state, action, reward, next_state, done)
        states.append(state)
        actions.append([action])
        rewards.append([reward])
        next_states.append(next_state)
        dones.append([done])
        state = next_state                             # roll over the state to next time step
        j += 1
        if done:                                       # exit loop if episode finished
            break

    # replay buffer size
    # maximum score
    # earliest acceptance criteria reached
    # ddqn - change
    # dueling
    # prioritized experience replay
    total_scores.append(score)
    if len(total_scores) > 100:
        total_scores = total_scores[(len(total_scores) - 100):]
    avg_score = float(sum(total_scores))/float(len(total_scores))
    with open('data-test.csv', 'a+') as f:
        f.write("{},{},{},{}\n".format(i, score, eps, avg_score))
    # env.reset(train_mode=True)
    env.reset()


agent.save_model()

