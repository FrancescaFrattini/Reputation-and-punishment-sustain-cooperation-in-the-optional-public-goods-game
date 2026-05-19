import itertools
import logging
import random

import numpy as np
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
    "alpha", "discount_factor", "q_table", "current_state", "n", 
    ]

    ACTIONS = [0, 1, None]  # Actions: cooperate (1), defect (0), withdraw (None)

    def __init__(self, ID, strategy, group_size, alpha=0.1, discount_factor=0.1, n=1):
        super().__init__(ID, strategy=strategy)
        self.alpha, self.discount_factor = alpha, discount_factor

        self.q_table = {}
        self.current_state = None
        self.n = n

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
        if random.random() < epsilon or self._is_uninitialized():
            self.tracker.append(random.choice(self.ACTIONS))
        else:
            best_action_index = np.argmax(self._get_row_values())
            self.tracker.append(self.ACTIONS[best_action_index])
        return self.tracker[-1]
    
    def learn(self, reward, avg):
        """        
        Updates the agent's Q-values based on the average payoff of the belonging group for the current round.
        At each time step, the agent observes the average payoff of the group for each possible action (1, 0, None).
        After each round, the agent updates its Q-values based on the observed value.
        Args:
            avg (float): The average payoff of the group for the current round.
            reward (float): The reward received after taking the last action.    
    """
        action_idx = self._action_to_index(self.tracker[-1])
        if avg not in self.q_table:
            #self.q_table[avg] = [0, 0, 0]
            self.q_table[avg] = self._init_state()
    
        if self.current_state is not None:
            current_q_value = self.q_table[self.current_state][action_idx][-1]
            next_q_value = self._get_max_q_value(avg)
            td_error = reward  + (self.discount_factor * next_q_value) - current_q_value
            new_q_value = current_q_value + (self.alpha * td_error)
            self.q_table[self.current_state][action_idx].append(new_q_value)
        
        self.current_state = avg
    
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

    def _get_row_values(self): return [max(deq) for deq in self.q_table[self.current_state]]

    def _init_state(self):
        return [
            deque([0], maxlen=self.n),
            deque([0], maxlen=self.n),
            deque([0], maxlen=self.n)
        ]

    def _get_max_q_value(self, state):
        return max(deq[-1] for deq in self.q_table[state])

    def _is_uninitialized(self):
        return all(len(deq) == 1 and deq[0] == 0 for deq in self.q_table[self.current_state])

