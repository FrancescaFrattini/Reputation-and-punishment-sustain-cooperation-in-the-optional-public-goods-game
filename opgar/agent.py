import itertools
import logging
import random

import numpy as np

from opgar.utils import Utils
from .strategy import _Strategy
from collections import deque, Counter

class _Agent:
    __slots__ = ["ID", "strategy", "utility", "reputation", "tracker"]

    def __init__(self, ID, strategy):
        self.ID = ID
        self.strategy = {
            key: val
            for key, val in zip(["behavioural", "punishment"], strategy.split("_"))
        }
        self.strategy["ID"] = strategy
        self.utility = 1
        self.reputation = 1
        self.tracker = deque(maxlen=10)
        
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
        self.tracker.append(_Strategy._choose_action(
            self.strategy["behavioural"], average_reputation
        ))
        return self.tracker[-1]

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
    "alpha", "discount_factor", "q_table", "current_state", "n", "state_to_idx", "idx_to_state"
    ]

    ACTIONS = [0, 1, None]  # Actions: cooperate (1), defect (0), withdraw (None)

    def __init__(self, ID, strategy, group_size, alpha=0.1, discount_factor=0.1, n=1):
        super().__init__(ID, strategy=strategy)
        self.alpha, self.discount_factor, self.n = alpha, discount_factor, n
        combinations = [
            (d, c, l)
            for d in range(group_size)
            for c in range(group_size)
            for l in range(group_size)
            if d + c + l == group_size - 1
        ]
        self.state_to_idx = {state: i for i, state in enumerate(combinations)}
        self.idx_to_state = dict(enumerate(combinations))
        self.q_table = np.empty((len(combinations), len(self.ACTIONS)), dtype=object)

        for i in range(len(combinations)):
            for j in range(len(self.ACTIONS)):
                self.q_table[i, j] = deque([0], maxlen=self.n)
        self.current_state = None
        

    def _choose_action(self, epsilon, average_reputation=None):
        """
        For Q-Learning agents there are two options for action selection:
        1. Epsilon-greedy action selection: with probability epsilon, or if at the first step, choose a random action
            Epsilon value decays at every time step of a factor of 0.99, ensuring exploration
        2. Greedy action selection: choose the action with the highest Q-value for the current state

        Args:
            epsilon (float): The exploration rate, between 0.05 and 1
            
        Returns:
            Action (str): Contributes 1 or 0 if playing, if not participating, then return None
    """
        if self.current_state is None or random.random() < epsilon or self._is_uninitialized():
            self.tracker.append(random.choice(self.ACTIONS))
        else:
            best_action_index = np.argmax(self._get_row_values(self.current_state))
            self.tracker.append(self.ACTIONS[best_action_index])
        return self.tracker[-1]
    
    def learn(self, reward, contributions):
        """        
        Updates the agent's Q-values based on the count of actions taken from other group's members.
        At each time step, the agent observes the contributions of other agents in the group for each possible action (1, 0, None).
        The contributions are stored in a circular buffer (self.q_table) of size n, where n is the number of time steps to remember.
        After each round, the agent updates its Q-values based on the observed contributions.
        Args:
            contributions (dict): A dictionary with keys as actions (1, 0, None) and values as the count of agents who chose 
            that action.
            reward (float): The reward received after taking the last action.    
    """
        
        triple = Utils.from_counter_to_tuple(counter = contributions)
        idx = self.state_to_idx[triple]
        action_idx = self._action_to_index(self.tracker[-1])

        if self.current_state is not None:
            td_error = reward  + (self.discount_factor*self._get_max_q_value(idx)) - self.q_table[self.current_state, action_idx][-1]
            if self.n == 1:
                self.q_table[self.current_state, action_idx] += self.alpha * td_error
            else:   
                self.q_table[self.current_state, action_idx].append(self.alpha * td_error)

        self.current_state = idx
    

    def _action_to_index(self, action):
            """
            Converts action (0, 1, None) to index (0, 1, 2) for Q-table access.
            Args:
                action (int or None): The action taken by the agent (0, 1, or None).
            Returns:
                int: The corresponding index in the Q-table (0 for action 0, 1 for action 1, 2 for action None).    
        """
            if action is None:  
                return 2
            else:
                return action

    def _get_row_values(self, state_idx): return [max(deq) for deq in self.q_table[state_idx]]

    def _get_max_q_value(self, state_idx):
        return max(
            value
            for action_deque in self.q_table[state_idx]
            for value in action_deque
        )

    def _is_uninitialized(self):
        return all (
            len(deq) == 1 and deq[0] == 0
            for row in self.q_table
            for deq in row
        )

