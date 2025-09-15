import logging
import random

import numpy as np
from .strategy import _Strategy
from collections import deque, Counter

class _Agent:
    __slots__ = ["ID", "strategy", "utility", "reputation", "tracker", "n"]

    def __init__(self, ID, strategy, n=10):
        self.ID = ID
        self.strategy = {
            key: val
            for key, val in zip(["behavioural", "punishment"], strategy.split("_"))
        }
        self.strategy["ID"] = strategy
        self.utility = 1
        self.reputation = 1
        self.tracker = deque(maxlen=n)
        self.n = n
        """
        logging.info(
            f"Agent {self.ID} created with r={self.reputation} & s={self.strategy}"
        ) """

    def _choose_action(self, average_reputation, epsilon=None):
        """
        _Agent chooses it's action in a public good game given the average reputation of the other players. Choice
        depends on the player's strategy.
        Args:
            average_reputation (float): The average reputation in [-1, 1] of the other players in the group

        Returns:
            Action (str): Contributes 1 or 0 if playing, if not participating, then return None
        """
        return _Strategy._choose_action(
            self.strategy["behavioural"], average_reputation
        )

    def choose_punishment(self, opponent_action):
        """
        _Agent chooses it's punishment for his opponent depending on his own punishment strategy and the opponent's
        previous action.
        Args:
            opponent_action ([1, 0, None]): Opponent's last action of cooperation/defection/withdrawal from the game.
        Returns:
            punishment (bool): Returns True/False depending on whether the agent punishes his opponent or not.
        """
        return _Strategy._choose_punishment(
            self.strategy["punishment"], opponent_action
        )

    def __str__(self):
        return (
            f"A(ID={self.ID}, "
            f"s={self.strategy['behavioural']}_{self.strategy['punishment']}, "
            f"u={round(self.utility, 4)}, "
            f"r={self.reputation})"
        )
    
class QLearningAgent(_Agent):

    __slots__ = _Agent.__slots__ + [
    "q_values", "alpha", "discount_factor", "q_table", "pos", "full","n"
    ]

    ACTIONS = [1, 0, None]  # Actions: cooperate (1), defect (0), withdraw (None)

    def _action_to_index(self, action):
        if action is None:  
            return 2
        else:
            return action


    def __init__(self, ID, strategy, alpha=0.1, discount_factor=0.1, n=10):
        super().__init__(ID, strategy=strategy, n=n)
        self.alpha, self.discount_factor = alpha, discount_factor
        self.q_table = np.zeros((n, len(self.ACTIONS)), dtype=np.int16)
        self.pos = 0
        self.full = False

    def _choose_action(self, epsilon, average_reputation=None):
        """
        For Q-Learning agents there are two options for action selection:
        1. Epsilon-greedy action selection: with probability epsilon, or if at the first step, choose a random action
            Epsilon value decays at every time step of a factor of 0.99, ensuring exploration
        2. Greedy action selection: choose the action with the highest Q-value

        Args:
            average_reputation (float): The average reputation in [-1, 1] of the other players in the group
            epsilon (float): The exploration rate, between 0.05 and 1
            
        Returns:
            Action (str): Contributes 1 or 0 if playing, if not participating, then return None
    """
        if random.random() < epsilon or not np.any(self.q_table):
            return random.choice(self.ACTIONS)
        
        sums = np.sum(self.q_table, axis=0)   
        best_action_index = np.argmax(sums)
        return self.ACTIONS[best_action_index]
    """
    def learn(self, reward, action_taken):
        
        Updates the agent's Q-values based on the action taken and the received reward.
            The Q-value for the action taken is updated using the Q-learning formula:
            Q(a) = Q(a) + α * (reward + γ * max(Q(a')) - Q(a))
            where:
            - Q(a) is the current Q-value for the action taken  
            - α is the learning rate
            - reward is the immediate reward received for the action taken
            - γ is the discount factor
            - max(Q(a')) is the maximum Q-value for the available actions, as there are no states here   
        Args:
            reward (float): The reward received for the action taken
            action_taken (str): The action taken by the agent, one of [1, 0, None]

        Returns: 
            None      
        
        idx = self.ACTIONS.index(action_taken)
        #as there are no states, we multiply the discount factor by the max Q-value
        td_error = reward  + (self.discount_factor*np.max(self.q_values)) - self.q_values[idx]      
        td_error = reward  + (self.discount_factor*np.max(self.q_values)) - self.q_values[idx]    
        self.q_values[idx] += self.alpha * td_error
    """
    def learn(self, contributions):
        for action in self.ACTIONS:
            self.push(action, contributions[action])
        self.pos = (self.pos + 1) % self.n
        if self.pos == 0:
            self.full = True

    '''
        Insert in the circular buffer last agent's contribution, overriding the oldest one.
    '''
    def push(self, action, value: int):
        idx = self._action_to_index(action)
        self.q_table[self.pos, idx] = value

    '''
      Return all elements in the circular buffer in correct order.
    '''
    def getLastRow(self):
        if self.pos == 0 and not self.full:
            return None
        last_index = (self.pos - 1) % self.n
        return self.q_table[last_index]
    
    """
    Return the last element inserted in the circular buffer.
    """
    def getLast(self, action):
        if self.pos == 0 and not self.full:
            return None
        last_index = (self.pos - 1) % self.n
        idx = self.action_to_index(action)
        return self.q_table[last_index, idx]