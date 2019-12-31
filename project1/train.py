from unityagents import UnityEnvironment
from tqdm import tqdm
import numpy as np
import torch
from torch import FloatTensor, LongTensor, cuda
import sys

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
env_info = env.reset(train_mode=True)[brain_name]

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


from pycode import QNetwork, Agent, ReplayBuffer

agent = Agent(state_size=state_size, action_size=action_size, seed=43)

env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
env_info
j = 0
eps = 1.
eps_step = 15
mx = 1800

replay_buffer_size = 10*200
total_scores = []
total_states, total_actions, total_rewards, total_next_states, total_dones = [[], [], [], [], []]
improvement = False

with open('data.csv', 'w') as f:
    f.write("Index,Score,Exploration,Rolling avg score\n")

stop = 0
max_value = -10  # start with negative score to prevent early training stop

for i in tqdm(range(mx)):
    states, actions, rewards, next_states, dones = [[], [], [], [], []]
    score = 0                                          # initialize the score
    while True:
        # action = np.random.randint(action_size)        # select an action
        action = agent.act(state, eps)
        # print(action)
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

    if i <= eps_step:
        eps = 0.6
    elif i <= 2*eps_step:
        eps = 0.5
    elif i <= 3*eps_step:
        eps = 0.4
    elif i <= 4*eps_step:
        eps = 0.3
    elif i <= 5*eps_step:
        eps = 0.2
    elif i <= 6*eps_step:
        eps = 0.1
    elif i <= 7*eps_step:
        eps = 0.05
    else:
        eps = 0.

    s_b, a_b, r_b, ns_b, d_b = [[], [], [], [], []]
    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        s_b.append(s)
        a_b.append(a)
        r_b.append(r)
        ns_b.append(ns)
        d_b.append(d)

        if len(s_b) > 10:
            s_b.pop(0)
            a_b.pop(0)
            r_b.pop(0)
            ns_b.pop(0)
            d_b.pop(0)

        if r != 0:
            total_states += s_b
            total_actions += a_b
            total_rewards += r_b
            total_next_states += ns_b
            total_dones += d_b

        if len(total_states) > replay_buffer_size:
            total_states = total_states[(len(total_states) - replay_buffer_size):]
            total_actions = total_actions[(len(total_actions) - replay_buffer_size):]
            total_rewards = total_rewards[(len(total_rewards) - replay_buffer_size):]
            total_next_states = total_next_states[(len(total_next_states) - replay_buffer_size):]
            total_dones = total_dones[(len(total_dones) - replay_buffer_size):]

    # replay buffer size
    # maximum score
    # earliest acceptance criteria reached
    # ddqn - change
    # dueling
    # prioritized experience replay

    if DEVICE == torch.device(type='cpu'):
        agent.learn((FloatTensor(states),
                     LongTensor(actions),
                     FloatTensor(rewards),
                     FloatTensor(next_states),
                     FloatTensor(dones)),
                    (1.-(1./action_size)))
        agent.learn((FloatTensor(total_states),
                     LongTensor(total_actions),
                     FloatTensor(total_rewards),
                     FloatTensor(total_next_states),
                     FloatTensor(total_dones)),
                    (1.-(1./action_size)))
    else:
        agent.learn((cuda.FloatTensor(states),
                     cuda.LongTensor(actions),
                     cuda.FloatTensor(rewards),
                     cuda.FloatTensor(next_states),
                     cuda.FloatTensor(dones)),
                    (1.-(1./action_size)))
        agent.learn((cuda.FloatTensor(total_states),
                     cuda.LongTensor(total_actions),
                     cuda.FloatTensor(total_rewards),
                     cuda.FloatTensor(total_next_states),
                     cuda.FloatTensor(total_dones)),
                    (1.-(1./action_size)))
    total_scores.append(score)
    if len(total_scores) > 100:
        total_scores = total_scores[(len(total_scores) - 100):]
    avg_score = float(sum(total_scores))/float(len(total_scores))
    with open('data.csv', 'a+') as f:
        f.write("{},{},{},{}\n".format(i, score, eps, avg_score))

    # print("Score: {}, i: {}, eps: {}, avg: {}".format(score, i, eps, avg_score))
    env.reset(train_mode=True)
    if not improvement and avg_score > 13.:
        print('Training completed in {} episodes.'.format(i))
        improvement = True
        # break

    if avg_score > max_value:
        max_value = avg_score
        stop = i

    if i > 200 and i > stop + 50:  # i > 200 don't consider stopping criteria if still exploring
        print('Training finished no improvements in score recorded in 50 episodes')
        break


agent.save_model()
