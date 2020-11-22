# Udacity--Deep-Reinforcement-Learning-Nano-Degree
This repository contains all the project codes related to Udacity Deep Reinforcement Learning Nano Degree

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

![Trained Agents][image1]

## Project 1
The goal of the project is to demonstrate the abilities of a model-free reinforcement learning algorithm, particularly Deep Q-Learning. The project uses Unity environment, a game development framework and Pytorch, a deep learning framework, to train an agent to solve an environment consisting of 37 continuous states and 4 actions. The goal of the agent is to maximize expected cumulative reward by collecting only yellow banana's (+1 reward) and avoiding blue banana's (-1 reward).

### Deep Q-Learning Algorithm
- Initialize replay memory D with capacity N
- Initialize action-value function q' with random weights w
- Initialize target action-value weights w' <-- w
- For the episode e <-- 1 to M:
  - Initial input frame X1
  - Prepare initial state: S <-- Z(<X1>)
  - For time step t <-- 1 to T:
    - Choose Action A from State S using policy pi <-- epsilon-greedy(q'(S,A,w))
    - Take action A, observer reward R and next input frame X2
    - Prepare next state S' <-- Z(<X-1,X0,X1,X2>)
    - Store expereince tuple (S,A,R,S') in replay memory D
    - S <-- S'
    - Obtain random minibatch of tuples (Sj,Aj,Rj,Sj+1) from D
    - Set target Yj = Rj + gamma*maxa(q'(Sj+1,Aj,w'))
    - Update deltaW = alpha * (Yj - q'(Sj,Aj,w))*deltaW(q'(Sj,Aj,w))
    - Every C steps, reset w' <-- w


![](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

## Project 2
The goal of the project is to demonstrate the abilities of a model-free reinforcement learning algorithm, particularly Deep Deterministic Policy Gradients (DDPG) Algorithm, which consists of two neural networks namely Actor network and Critic network. The project uses Unity environment, a game development framework and Pytorch, a deep learning framework. The algorithm was trained on 20 agents, with each agent consisting of 33 states and 4 actions.

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

## Project 3
The goal of the project is to demonstrate the abilities of a model-free reinforcement learning algorithm, particularly Multi-Agent Deep Deterministic Policy Gradients (MADDPG) Algorithm, which consists of four neural networks for each of the agents, namely Actor and Critic neural networks. The project uses Unity environment, a game development framework and Pytorch, a deep learning framework. The algorithm was trained on 2 agents, with each agent consisting of 24 states and 2 action/observation spaces.

![](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)
