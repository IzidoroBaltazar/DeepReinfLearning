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
mx = 10000

improvement = False
total_scores = []

with open('data.csv', 'w') as f:
    f.write("Index,Score,Rolling avg score\n")

stop = 0
max_value = -10  # start with negative score to prevent early training stop
num_agents = len(env_info.agents)
agent = Agent(state_size=state_size, action_size=action_size, seed=2276, num_agents=num_agents)

all_states, all_actions, all_rewards, all_next_states, all_dones = [[], [], [], [], []]
t = trange(1, mx+1)
for i in t:
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations             # get the current state
    # states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    agent.reset()

    while True:
        # for state in states:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states

        if np.any(dones):
            break

    total_scores.append(np.mean(scores))
    avg_score = np.mean(total_scores[i-min(i,100):i+1])
    with open('data.csv', 'a+') as f:
        f.write("{},{:.3f},{:.3f}\n".format(i, total_scores[-1], avg_score))

    # print("Score: {:.3f}, i: {}, avg: {:.3f}".format(scores[0], i, avg_score))
    env.reset(train_mode=True)
    if not improvement and avg_score > 30.:
        print('Training completed in {} episodes.'.format(i))
        improvement = True
        # break

    # t.set_description('Score {:.3f}, Score Avg. {:.3f}'.format(scores[0], avg_score))
    t.set_postfix(score=total_scores[-1], score_avg=avg_score)

    if avg_score > max_value:
        agent.save_model()
        max_value = avg_score
        stop = i

    if i > 4000 and i > stop + 80:  # i > 200 don't consider stopping criteria if still exploring
        print('Training finished no improvements in score recorded in 80 episodes')
        break
