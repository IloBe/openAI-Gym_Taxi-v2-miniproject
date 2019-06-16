# openAI-Gym_Taxi-v2-miniproject
Reinforcement learning: we teach a taxi agent to navigate through its gridworld by using the OpenAI Gym's Taxi-v2 environment.


<h2>1. Technical Information</h2>

OpenAI Gym environment: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

The workspace contains 3 Python files, including the implemented agent:
- agent.py: The reinforcement learning concept the agent reacts on with hyperparameters. 
- monitor.py: The interact function tests how well the agent learns from interaction with the environment. 
- main.py: This file is the starting point for a terminal run and checks the performance of the agent. 

To start the learning via terminal, change to the associated directory that includes these files and execute it by typing:<br>python main.py


<h2>2. Implementation Concept</h2>

Description of the agent implementation concept:<br>
To solve this task 2 Temporal-Difference methods, TD learning for short, have been tested.<br>
The general idea behind this TD methods is, that during learning the agent does not have to wait until episodes end, having the final update values. With TD methods prediction improvements are created at every step.

First, the agent has been implemented with SARSA and evaluated with several hyperparameters, starting with random values. Then after getting an intuition and having a look to the evaluated performance values, the implementation has changed to Expected SARSA. 
The acronym SARSA means, each action value update uses a state-action-reward, next state, next action tuple of interaction.

'Sarsa is guaranteed to converge to the optimal action-value function q∗ (and so yield the optimal policy π∗), as long as the step-size parameter α is sufficiently small, and the Greedy in the Limit with Infinite Exploration (GLIE) conditions are met.' (Udacity ML part 2 text)<br>
For that an ϵ-greedy policy is implemented.<br>
Both methods, SARSA and Expected SARSA, are on-policy TD control algorithms. Means, the same (ϵ-greedy) policy that is evaluated and improved is also used to select actions.

Regarding Expected SARSA:<br>
Its specific behaviour is, that the expected value of the next state-action pair is choosen and not the maximum as it happened with SARSAMAX (Q-learning). Means, probabilities are taken into account that the agent selects each possible action from the next state.

The agent interacts with the environment for 20,000 episodes. The details of the interaction are specified in monitor.py which returns two variables:
- avg_rewards: a deque where avg_rewards[i] is the average return collected by the agent from episodes i+1 to episode i+100, inclusive.
- best_avg_reward: final, largest score from avg_rewards used to identify the best task performance of the agent.

The much better performance results of Expected SARSA leads to the decision to store only its agent Python implementation in this repository.


<h2>3. Performance</h2>

Performance evaluation results regarding the best average reward:<br>
Hyperparameter - result

<h3>3.1 Sarsa implementation</h3>

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


<h3>3.2 Expected Sarsa implementation</h3>

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


<h3>3.3 Best Result</h3>

Final configuration with Extended SARSA:<br>
alpha=0.02, gamma=1.0, epsilon=0.001
