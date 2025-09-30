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

    ACTIONS = [0, 1, None]  # Actions: cooperate (1), defect (0), withdraw (None)

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
        logging.info(f"Agent {self.ID} choosing best action with Q-values sums: {self.ACTIONS[best_action_index]}")
        return self.ACTIONS[best_action_index]
    
    """        
        Updates the agent's Q-values based on the count of actions taken from other group's members.
        At each time step, the agent observes the contributions of other agents in the group for each possible action (1, 0, None).
        The contributions are stored in a circular buffer (self.q_table) of size n, where n is the number of time steps to remember.
        After each round, the agent updates its Q-values based on the observed contributions.
        Args:
            contributions (dict): A dictionary with keys as actions (1, 0, None) and values as the count of agents who chose 
            that action.
        Returns: 
            None      
    """
    def learn(self, contributions):
        for action in self.ACTIONS:
            self.push(action, contributions[action])
        self.pos = (self.pos + 1) % self.n
        if self.pos == 0:
            self.full = True
        logging.info(f"group's contribution: {contributions}")
        logging.info(f"updated Q-Table for agent {self.ID}:\n {self.q_table}")
        logging.info(f"q-table position index {self.pos}")

    '''
        Insert in the circular buffer last agent's contribution, overriding the oldest one.
        Args:
            action (str): one from [1, 0, None]
            value (int): Number of group's agents who chose that action
        Returns:
            None
    '''
    def push(self, action, value: int):
        idx = self._action_to_index(action)
        self.q_table[self.pos, idx] = value

    '''
      Return all elements of the circular buffer in the last row
      Returns:
        Array of shape (1, 3) containing the last contribution for each action.
    '''
    def getLastRow(self):
        if self.pos == 0 and not self.full:
            return None
        last_index = (self.pos - 1) % self.n
        return self.q_table[last_index]
    
    """
    Return a single cell of the circular buffer for the given action in the last row.
    Args:
        action (str): one from [1, 0, None]
    Return:
        Int value of the last contribution for the given action.
    """
    def getLast(self, action):
        if self.pos == 0 and not self.full:
            return None
        last_index = (self.pos - 1) % self.n
        idx = self.action_to_index(action)
        return self.q_table[last_index, idx]