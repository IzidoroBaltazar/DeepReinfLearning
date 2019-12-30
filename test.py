from unityagents import UnityEnvironment
import numpy as np
import torch
from torch import FloatTensor, LongTensor, cuda


# env = UnityEnvironment(file_name="./Banana_Linux")
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
# env = UnityEnvironment(file_name="./Banana.app")

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
mx = 1800

total_states, total_actions, total_rewards, total_next_states, total_dones = [[], [], [], [], []]

for i in range(mx):
    states, actions, rewards, next_states, dones = [[], [], [], [], []]
    score = 0                                          # initialize the score
    while True:
        # action = np.random.randint(action_size)        # select an action
        action = agent.act(state, eps)
        # print(action)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        # stp = env.step(action)
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
        # print(env_info.visual_observations)
        # if i > 10:
        #     break
        j += 1
        # if j == 3:
        #     break
        if done:                                       # exit loop if episode finished
            break

    if i <= (mx/4):
        eps = 0.5
    elif i <= (mx/2):
        eps = 0.25
    elif i <= 3*(mx/4):
        eps = 0.1
    else:
        eps = 0.

    s_b, a_b, r_b, ns_b, d_b = [[], [], [], [], []]
    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        s_b.append(s)
        a_b.append(a)
        r_b.append(r)
        ns_b.append(ns)
        d_b.append(d)

        if len(s_b) > 5:
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

    # print(actions.unique)
    # print(FloatTensor(actions).size)
    # print(FloatTensor(states).size)
    # print(FloatTensor(rewards))
    # print(FloatTensor(next_state).size)
    # print(FloatTensor(dones).size)
    agent.learn((cuda.FloatTensor(states),
                 cuda.LongTensor(actions),
                 cuda.FloatTensor(rewards),
                 cuda.FloatTensor(next_states),
                 cuda.FloatTensor(dones)),
                (1.-(1./action_size)))
    # agent.learn((cuda.FloatTensor(total_states),
    #              cuda.LongTensor(total_actions),
    #              cuda.FloatTensor(total_rewards),
    #              cuda.FloatTensor(total_next_states),
    #              cuda.FloatTensor(total_dones)),
    #             (1.-(1./action_size)))

    print("Score: {}, i: {}, eps: {}".format(score, i, eps))
    env.reset(train_mode=True)

agent.save_model()

# agent.learn((FloatTensor(states), LongTensor(actions), FloatTensor(rewards), FloatTensor(next_states), LongTensor(dones)), (1.-(1./action_size)))

