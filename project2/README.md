# Project 2
Deep Reinforcement Learning Udacity Nanodegree

## Task
Navigate robot arm to move into target sphere on the rewards awarded using deep reinforcement learning.

### Input
Vector of size 33.

### Rewards
Based on the time spent in the target sphere.

### Goal
Reach average reward of at least 30 points per episode.

## Solution
We used code from previous code example as Deep Deterministic Policy Gradient learning [repo link](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal).

### Parameters
* `mx` - number of maximal training iterations `1000`

### Modification of nosie generator
Code below is highlighting change of noise generator from the original `ddpg-bipedal` example.
```python
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    ...
        """Sigma reduction"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)
```
### Training
Without the modification of Ornstein-Uhlenbeck process I was not able to reach the training goal.
Neural network configuratio is same as in `ddpg-bipedal` 1st layer contains 256 neurons and 2nd layer has 128 neurons.
Activation function is hyperbolic tangens because action is vector of size 4 and values are continuous from -1 to 1.
Model was saved every time new rolling average maximum was reached.
If no new rolling average maximum was reached in 80 episodes traning was terminated.

To reproduce achieved results parameters you can run `python3 train.py` (assuming all of the `requirements.txt` are fullfilled).
Image showing solution to the problem. Robot arm follows taget.
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project2/test.gif)

* Objective was reached in 309 episodes.
* Maximum rolling average was reached in 368 episodes.
* Max rolling average: 37.159
* Max score: 39.63
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project2/figure-train.png)

### Test
* Test run with model weights loaded from `model/weights_local.torch` and `model/weights_target.torch` weights for cirtic are stored and loaded as well.
* To test included model you can run `python3 test.py` it will generate file data-test.csv with performace data recorded.
* Max rolling average: 38.56
* Max score: 39.47
![alt text](https://raw.githubusercontent.com/IzidoroBaltazar/DeepReinfLearning/master/project2/figure-test.png)

### Conclusions
Model training was very sensitive to changes in noise or qnetwork architecture modifications.
However network could be trained for the target score with wide range of number of neurons.
