# openAI-Gym_Taxi-v2-miniproject
A reinforcement learning examination: we teach a taxi agent to navigate through its gridworld by using the OpenAI Gym's Taxi-v2 environment.


## 1. Technical Information

The used OpenAI Gym environment: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

The workspace contains 3 Python files, including the implemented agent:
- agent.py: The reinforcement learning concept the agent reacts on with hyperparameters. 
- monitor.py: The interact function tests how well the agent learns from interaction with the environment. 
- main.py: This file is the starting point for a terminal run and checks the performance of the agent. 

To use such coding, the Python _.py_ files have to be downloaded. They are implemented with Python version 3.6 which has to be installed.
Furthermore, a common terminal tool shall be used for running the code.
To start the learning via terminal, change to the associated directory that includes these files and execute it by typing:
```
python main.py
```


## 2. Implementation Concept

### 2.1 General Information

Description of the **agent implementation** concept:<br>
To solve this task 2 Temporal-Difference methods, TD learning for short, have been tested:
- SARSA
- Expected SARSA

The acronym SARSA means, each action value update uses a state-action-reward, next state, next action tuple of interaction.

The general idea behind this TD methods is, that during learning the agent does not have to wait until episodes end, having the final update values. With TD methods prediction improvements are created at every step.

First, the agent has been implemented with SARSA and has been evaluated with several hyperparameters, started with random values. Then after having got an intuition with the evaluated performance values, the implementation has been changed to Expected SARSA. 

'Sarsa is guaranteed to converge to the optimal _action-value function q∗_ (and so yield the _optimal policy π∗_), as long as the step-size parameter α is sufficiently small, and the Greedy in the Limit with Infinite Exploration (GLIE) conditions are met.' (Udacity ML part 2 text)<br>
For that an _ϵ-greedy policy_ has been implemented in the agent class.<br>
Both methods, SARSA and Expected SARSA, are on-policy TD control algorithms. Means, the same (ϵ-greedy) policy that is evaluated and improved is also used to select actions.

Regarding Expected SARSA:<br>
Its specific behaviour is, that the expected value of the next state-action pair is choosen and not the maximum as it happened with SARSAMAX (Q-learning). Means, probabilities are taken into account that the agent selects each possible action from the next state.

### 2.2 Implementation Details

The agent interacts with the environment for 20,000 episodes. The details of the interaction are specified in monitor.py which returns two variables:
- avg_rewards: a deque where avg_rewards[i] is the average return collected by the agent from episodes i+1 to episode i+100, inclusive.
- best_avg_reward: final, largest score from avg_rewards used to identify the best task performance of the agent.

The much better performance results of Expected SARSA leads to the decision to store only its agent Python implementation in this repository.


## 3. Performance

Performance evaluation results regarding the best average reward:<br>
Hyperparameter - result

### 3.1 Sarsa implementation

alpha=0.5, gamma=0.9, epsilon=0.1 - Best average reward -89.75<br>
alpha=0.5, gamma=0.9, epsilon=0.1 - Best average reward -91.01<br>
alpha=0.1, gamma=0.9, epsilon=0.1 - Best average reward -10.49<br>
alpha=0.1, gamma=0.9, epsilon=0.1 - Best average reward -09.01<br>
alpha=0.1, gamma=0.9, epsilon=0.05 - Best average reward -09.933<br>
alpha=0.01, gamma=0.9, epsilon=0.05 - Best average reward -10.84<br>
alpha=0.01, gamma=0.9, epsilon=0.1 - Best average reward -26.44<br>
alpha=0.05, gamma=0.9, epsilon=0.05 - Best average reward -0.755<br>
alpha=0.05, gamma=0.7, epsilon=0.05 - Best average reward -88.867<br>
alpha=0.05, gamma=1.0, epsilon=0.05 - Best average reward 3.9177<br>
alpha=0.04, gamma=1.0, epsilon=0.05 - Best average reward 2.8448<br>
alpha=0.06, gamma=1.0, epsilon=0.05 - Best average reward 5.2119<br>
alpha=0.06, gamma=1.0, epsilon=0.05 - Best average reward 4.0571<br>
alpha=0.04, gamma=1.0, epsilon=0.05 - Best average reward 3.8729<br>
alpha=0.065, gamma=1.0, epsilon=0.045 - Best average reward 5.4585<br>
alpha=0.065, gamma=1.0, epsilon=0.045 - Best average reward 4.9616<br>
alpha=0.065, gamma=1.0, epsilon=0.04 - Best average reward 5.0631<br>
alpha=0.065, gamma=1.0, epsilon=0.04 - Best average reward 4.4355<br>
alpha=0.063, gamma=1.0, epsilon=0.045 - Best average reward 5.2681<br>
alpha=0.063, gamma=1.0, epsilon=0.045 - Best average reward 4.7191


### 3.2 Expected Sarsa implementation

alpha=0.05, gamma=0.9, epsilon=0.045 - Best average reward 6.9585<br>
alpha=0.05, gamma=0.9, epsilon=0.045 - Best average reward 7.541<br>
alpha=0.05, gamma=0.9, epsilon=0.045 - Best average reward 7.7616<br>
alpha=0.055, gamma=0.9, epsilon=0.045 - Best average reward 7.225<br>
alpha=0.055, gamma=0.9, epsilon=0.045 - Best average reward 7.395<br>
alpha=0.045, gamma=0.9, epsilon=0.045 - Best average reward 7.656<br>
alpha=0.045, gamma=0.9, epsilon=0.045 - Best average reward 7.454<br>
alpha=0.045, gamma=0.9, epsilon=0.045 - Best average reward 7.311<br>
alpha=0.045, gamma=0.9, epsilon=0.05 - Best average reward 7.2371<br>
alpha=0.05, gamma=1.0, epsilon=0.05 - Best average reward 7.743<br>
alpha=0.05, gamma=1.0, epsilon=0.05 - Best average reward 7.675<br>
alpha=0.018, gamma=1.0, epsilon=0.001 - Best average reward 9.0754<br>
alpha=0.019, gamma=0.9, epsilon=0.001 - Best average reward 9.0442<br>
alpha=0.02, gamma=0.8, epsilon=0.0011 - Best average reward 8.9295<br>
alpha=0.02, gamma=0.9, epsilon=0.0011 - Best average reward 9.1916<br>
alpha=0.02, gamma=0.9, epsilon=0.001 - Best average reward 9.1358<br>
alpha=0.02, gamma=0.9, epsilon=0.001 - Best average reward 9.1355<br>
alpha=0.02, gamma=1.0, epsilon=0.001 - Best average reward 9.2565<br>
alpha=0.02, gamma=1.0, epsilon=0.001 - Best average reward 9.4292<br>
alpha=0.02, gamma=1.0, epsilon=0.001 - Best average reward 9.1764<br>
alpha=0.025, gamma=1.0, epsilon=0.001 - Best average reward 9.3117<br>
alpha=0.025, gamma=1.0, epsilon=0.001 - Best average reward 9.3169<br>
alpha=0.025, gamma=1.0, epsilon=0.001 - Best average reward 9.2471<br>
alpha=0.022, gamma=1.0, epsilon=0.001 - Best average reward 9.1979<br>
alpha=0.022, gamma=1.0, epsilon=0.001 - Best average reward 9.2247<br>
alpha=0.022, gamma=0.9, epsilon=0.001 - Best average reward 9.2927<br>
alpha=0.022, gamma=0.9, epsilon=0.001 - Best average reward 9.0545<br>
alpha=0.022, gamma=0.9, epsilon=0.001 - Best average reward 9.2244


### 3.3 Best Result

Final configuration with Extended SARSA:<br>
alpha=0.02, gamma=1.0, epsilon=0.001


## 4. License
This miniproject coding is released under the [MIT Licence](https://github.com/IloBe/openAI-Gym_Taxi-v2-miniproject/LICENCE).
