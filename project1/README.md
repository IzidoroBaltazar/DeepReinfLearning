# Project 1
Deep Reinforcement Learning Udacity Nanodegree

## Task
Navigate unknown environment based on the rewards awarded using deep reinforcement learning.

### Input
Vector of size 37.

### Rewards
* 0 - no banana was picked up
* 1 - yellow banana was picked up
* -1 - black banana was picked up

## Solution
I used code from previous example as dqn agent [repo link](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py)

I have recorded results below.

* Training with 20 episodes spent on each exploration level. Random action was taken with probability from 0.6 - 0.05.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure.png)
* Training with 15 episodes spent on each exploration level. Random action was taken with probability from 0.6 - 0.05. No data augmentation.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure-eps-step-15-no-augmentation.png)
* Training with 5 episodes spent on each exploration level. Random action was taken with probability from 0.6 - 0.05.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure-eps-step-5.png)
* Training with 15 episodes spent on each exploration level. Random action was taken with probability from 0.6 - 0.05.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure-eps-step-15.png)
* Training with 20 episodes spent on each exploration level. Random action was taken with probability from 0.6 - 0.05.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure-eps-step-20.png)
* Training with 20 episodes spent on each exploration level. Random action was taken with probability from 0.6 - 0.05. Model was trained on the selected buffer and each episode.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure-train-modified-buffer.png)
* Test run with model weights loaded from `model/weights_local.torch` and `model/weights_target.torch`.
* To test included model you can run `python test.py` it will generate file data-test.csv with performace data recorded.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project1/figure-test.png)
