import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, observation_space=500, alpha=0.02, gamma=1.0, epsilon=0.001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - other hyperparameters ...
        """
        self.nA = nA
        self.action_space = nA
        self.observation_space  = observation_space
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.0001
        self.epssilon_decay = 0.9999

        
    def create_epsilon_greedy_policy(self, Q_state):
        policy_state = np.ones(self.nA) * self.epsilon/self.nA # is a matrix, with epsilon/nA it would be only a float number
        max_action = np.argmax(Q_state)
        policy_state[max_action] = 1-self.epsilon + (self.epsilon/self.nA)    
        return policy_state

        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        policy_s = self.create_epsilon_greedy_policy(self.Q[state])
        self.epsilon = max(self.epsilon, self.epsilon_min)
        action_s = np.random.choice(np.arange(self.nA), p = policy_s)
        
        return action_s

    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # update TD estimate of Q 
        # by expected sarsa
        if done:
            # state-action pair and its value for Q; because done = 0, no next_state is available (so, gamma*0)
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
        else:
            # create epsilon-greedy action probabilities for the next state
            # pick next action A' with probabilities
            next_action = self.select_action(state)
            policy_s = self.create_epsilon_greedy_policy(self.Q[next_state])
            expected_value = np.dot(self.Q[next_state], policy_s)
            self.Q[state][next_action] += self.alpha*(reward+(self.gamma*expected_value)-self.Q[state][next_action])
            