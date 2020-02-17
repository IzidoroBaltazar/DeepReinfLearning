from unityagents import UnityEnvironment
from tqdm import trange
import numpy as np
import sys
from pycode import SoccerAgent

if sys.platform == "darwin":
    env = UnityEnvironment(file_name="./Tennis.app")
else:
    env = UnityEnvironment(file_name="./Soccer_Linux_NoVis/Soccer.x86_64")
    # env = UnityEnvironment(file_name="./Reacher_Linux_NoVis/Reacher.x86_64")

# set the goalie brain
g_brain_name = env.brain_names[0]
g_brain = env.brains[g_brain_name]

# set the striker brain
s_brain_name = env.brain_names[1]
s_brain = env.brains[s_brain_name]

print(s_brain_name)
print(s_brain)

state_size = s_brain.vector_observation_space_size
print("State size: ", state_size)

# reset the environment
env_info = env.reset(train_mode=True)

# number of agents
num_g_agents = len(env_info[g_brain_name].agents)
print('Number of goalie agents:', num_g_agents)
num_s_agents = len(env_info[s_brain_name].agents)
print('Number of striker agents:', num_s_agents)

# number of actions
g_action_size = g_brain.vector_action_space_size
print('Number of goalie actions:', g_action_size)
s_action_size = s_brain.vector_action_space_size
print('Number of striker actions:', s_action_size)

# examine the state space
g_states = env_info[g_brain_name].vector_observations
g_state_size = g_states.shape[1]
print('There are {} goalie agents. Each receives a state with length: {}'.format(g_states.shape[0], g_state_size))
s_states = env_info[s_brain_name].vector_observations
s_state_size = s_states.shape[1]
print('There are {} striker agents. Each receives a state with length: {}'.format(s_states.shape[0], s_state_size))

# env_info
j = 0
mx = 10**4

improvement = False
total_scores = []
total_scores_g = []
total_scores_s = []

with open('data-soccer.csv', 'w') as f:
    f.write("Index,Score_g,Score_s,Rolling avg score g,Rolling avg score s,Rolling avg score\n")

threshold = 0.1
stop = 0
max_value = -10  # start with negative score to prevent early training stop
g_num_agents = len(env_info[g_brain_name].agents)
s_num_agents = len(env_info[s_brain_name].agents)
agent = SoccerAgent(g_state_size=g_state_size, g_action_size=g_action_size,
                    seed=2277, num_agents=g_num_agents, s_state_size=g_state_size,
                    s_action_size=s_action_size, g_brain_name=g_brain_name,
                    s_brain_name=s_brain_name)

avg_train_score = 0
t = trange(1, mx+1)
for i in t:
    env_info = env.reset(train_mode=False)                 # reset the environment
    g_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
    s_states = env_info[s_brain_name].vector_observations  # get initial state (strikers)
    g_scores = np.zeros(num_g_agents)                      # initialize the score (goalies)
    s_scores = np.zeros(num_s_agents)                      # initialize the score (strikers)
    eps = 1/i

    while True:
        # select actions and send to environment
        # g_actions = np.random.randint(g_action_size, size=num_g_agents)
        # s_actions = np.random.randint(s_action_size, size=num_s_agents)
        # print('g_states.shape: ', g_states.shape)
        # g_actions = g_agent.act(g_states, eps=eps)
        # s_actions = s_agent.act(s_states, eps=eps)
        # print('g_actions:', g_actions)
        # print('s_actions:', g_actions)
        # actions = dict(zip([g_brain_name, s_brain_name],
        #                    [g_actions, s_actions]))
        actions = agent.act(g_states, s_states, eps=eps)
        env_info = env.step(actions)

        # get next states
        g_next_states = env_info[g_brain_name].vector_observations
        s_next_states = env_info[s_brain_name].vector_observations

        # get reward and update scores
        g_rewards = env_info[g_brain_name].rewards
        s_rewards = env_info[s_brain_name].rewards
        g_scores += g_rewards
        s_scores += s_rewards

        # check if episode finished
        # done = np.any(env_info[g_brain_name].local_done)
        done = np.any(env_info[g_brain_name].local_done)
        # import pdb; pdb.set_trace()

        # agent.step(**e)
        agent.step(
            g_states, actions[g_brain_name], g_rewards, g_next_states, [done, done],
            s_states, actions[s_brain_name], s_rewards, s_next_states,
            )

        # roll over states to next time step
        g_states = g_next_states
        s_states = s_next_states

        # exit loop if episode finished
        if done:
            break
    print('Scores from episode {}: {} (goalies), {} (strikers)'.format(i+1, g_scores, s_scores))

    # if True or np.max(scores) >= 0:
    #     j = 1
    #     # if avg_train_score == 0:
    #     #     avg_train_score = np.max(scores)
    #     # else:
    #     #     avg_train_score = (avg_train_score + np.max(scores)) / 2
    #     # if avg_train_score > threshold:
    #     #     threshold += 0.1
    #     # if np.max(scores) > 0:
    #     #     j = 2
    #     for _ in range(j):
    #         for e in episodes:
    #             agent.step(**e)
    # episodes = []

    total_scores.append(np.max(g_scores + s_scores))
    total_scores_g.append(np.max(g_scores))
    total_scores_s.append(np.max(s_scores))
    avg_score = np.mean(total_scores[i-min(i, 100):i+1])
    avg_score_g = np.mean(total_scores_g[i-min(i, 100):i+1])
    avg_score_s = np.mean(total_scores_s[i-min(i, 100):i+1])
    with open('data-soccer.csv', 'a+') as f:
        f.write("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(i, total_scores_g[-1], total_scores_s[-1],
                                            avg_score_g, avg_score_s, avg_score))

    env.reset(train_mode=True)
    if not improvement and avg_score > 30.:
        print('Training completed in {} episodes.'.format(i))
        improvement = True

    # t.set_postfix(threshold=threshold, score=total_scores[-1], score_avg=avg_score)
    # avg_score =

    if avg_score > max_value:
        agent.save_model()
        max_value = avg_score
        stop = i

    # if avg_score > 0.5:
    #     break

    if i > 1200 and i > stop + 200:  # i > 200 don't consider stopping criteria if still exploring
        print('Training finished no improvements in score recorded in 200 episodes')
        break
