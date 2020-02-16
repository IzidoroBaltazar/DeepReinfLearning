from unityagents import UnityEnvironment
from tqdm import trange
import numpy as np
import sys
from pycode import Agent

if sys.platform == "darwin":
    env = UnityEnvironment(file_name="./Tennis.app")
else:
    env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")
    # env = UnityEnvironment(file_name="./Reacher_Linux_NoVis/Reacher.x86_64")

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

# env_info
j = 0
mx = 10**6

improvement = False
total_scores = []

with open('data.csv', 'w') as f:
    f.write("Index,Score,Rolling avg score\n")

threshold = 0.1
stop = 0
max_value = -10  # start with negative score to prevent early training stop
num_agents = len(env_info.agents)
agent = Agent(state_size=state_size, action_size=action_size,
              seed=2277, num_agents=num_agents)

avg_train_score = 0
t = trange(1, mx+1)
for i in t:
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations             # get the current state
    scores = np.zeros(num_agents)
    agent.reset()
    episodes = []

    while True:
        # for state in states:
        actions = agent.act(states)
        # send all actions to tne environment
        env_info = env.step(actions)[brain_name]
        # get next state (for each agent)
        next_states = env_info.vector_observations
        # get reward (for each agent)
        rewards = env_info.rewards
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards
        episodes.append({'state': states,
                         'action': actions,
                         'reward': np.array(rewards),
                         'next_state': next_states,
                         'done': np.array(dones)})
        # agent.step(states, actions, np.array(rewards), next_states, np.array(dones))
        # print('states: ', states.shape)
        # print('actions: ', actions.shape)
        # print('rewards: ', rewards)
        # print('next_states: ', next_states.shape)
        # print('dones: ', dones)
        # print('states: ', states[0])
        # exit(0)
        states = next_states

        if np.any(dones):
            break

    if True or np.max(scores) >= 0:
        j = 1
        # if avg_train_score == 0:
        #     avg_train_score = np.max(scores)
        # else:
        #     avg_train_score = (avg_train_score + np.max(scores)) / 2
        # if avg_train_score > threshold:
        #     threshold += 0.1
        if np.max(scores) > 0:
            j = 2
        for _ in range(j):
            for e in episodes:
                agent.step(**e)
    episodes = []

    total_scores.append(np.max(scores))
    avg_score = np.mean(total_scores[i-min(i, 100):i+1])
    with open('data.csv', 'a+') as f:
        f.write("{},{:.3f},{:.3f}\n".format(i, total_scores[-1], avg_score))

    env.reset(train_mode=True)
    if not improvement and avg_score > 30.:
        print('Training completed in {} episodes.'.format(i))
        improvement = True

    t.set_postfix(threshold=threshold, score=total_scores[-1], score_avg=avg_score)

    if avg_score > max_value:
        agent.save_model()
        max_value = avg_score
        stop = i

    # if avg_score > 0.5:
    #     break

    if i > 2000 and i > stop + 1000:  # i > 200 don't consider stopping criteria if still exploring
        print('Training finished no improvements in score recorded in 1000 episodes')
        break
