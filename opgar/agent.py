import logging
import random

import numpy as np
from .strategy import _Strategy
from collections import deque

class _Agent:
    __slots__ = ["ID", "strategy", "utility", "reputation", "tracker"]

    def __init__(self, ID, strategy, n = 2):
        self.ID = ID
        self.strategy = {
            key: val
            for key, val in zip(["behavioural", "punishment"], strategy.split("_"))
        }
        self.strategy["ID"] = strategy
        self.utility = 1
        self.reputation = 1
        self.tracker = deque(maxlen=n)
        """
        logging.info(
            f"Agent {self.ID} created with r={self.reputation} & s={self.strategy}"
        ) """

    def _choose_action(self, average_reputation, groups_action_tracker=None):
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
    "qtable", "alpha", "discount_factor", "epsilon"
    ]

    ACTIONS_SIZE = 3

    minimum_epsilon = 0.01  # minimum exploration probability

    ACTIONS = [1, 0, None]  # Actions: cooperate (1), defect (0), withdraw (None)

    action_to_index = {1: 0, 0: 1, None: 2}
    index_to_action = {0: 1, 1: 0, 2: None}

    def __init__(self, ID, strategy, n=2, alpha=0.1, discount_factor=0.1, epsilon=1.0, state_size=1, action_size=3):
        super().__init__(ID, strategy=strategy, n=n)
        self.alpha, self.discount_factor, self.epsilon = alpha, discount_factor, epsilon
        #TODO fix state size and action size
        #self.state_size = state_size
        #self.action_size = action_size
        self.reset_qtable()

    def _state_to_index(self, state_tuple):
        """Converte una tupla di 4 azioni (dei compagni) in un indice [0, 80]"""
        idx = 0
        for i, a in enumerate(state_tuple):
            idx += self.action_to_index[a] * (self.ACTIONS_SIZE ** (self.ACTIONS_SIZE - i))
        return idx


    def _choose_action(self, average_reputation, state=None):
        """
        For Q-Learning agents there are two options for action selection:
        1. Epsilon-greedy action selection: with probability epsilon, choose action based on group's payoff, if all
            values are 0, choose randomly from the available actions.
            Epsilon value decays at every time step of a factor of 0.99, ensuring exploration
        2. Greedy action selection: choose the action with the highest Q-value

        Args:
            average_reputation (float): The average reputation in [-1, 1] of the other players in the group
            ignoring average_reputation for now, as this is a Q-Learning agent
        
        Returns:
            Action (str): Contributes 1 or 0 if playing, if not participating, then return None
        """
    
        if self.epsilon > self.minimum_epsilon:
            self.epsilon *= 0.99


        if random.random() < self.epsilon or state is None:
            '''
                group = next(g for g in groups_action_tracker if str(self.ID) in g)
                if not all(v == (0, 0) for v in group.values()) and group is not None:
                    return max(group.values(), key=lambda x: x[1])[0]
            '''
            return random.choice(self.ACTIONS)

        s_idx = self._state_to_index(state)
        logging.info(f"choosing next action by exploiting qtable, action chose is {self.index_to_action[np.argmax(self.qtable[s_idx])]}")

        return self.index_to_action[np.argmax(self.qtable[s_idx])]
        #return self.ACTIONS[int(np.argmax(self.q_values))]

    def learn(self, state, reward, action_taken, next_state):
        """
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
        """

        if state is None:
            return

        s_idx = self._state_to_index(state)
        a_idx = self.action_to_index[action_taken]
        ns_idx = self._state_to_index(next_state)

        td_target = reward + self.discount_factor * np.max(self.qtable[ns_idx])
        td_error = td_target - self.qtable[s_idx, a_idx]

        self.qtable[s_idx, a_idx] += self.alpha * td_error

        """
        idx = self.ACTIONS.index(action_taken)
        td_error = reward  + (self.discount_factor*np.max(self.q_values)) - self.q_values[idx]    
        self.q_values[idx] += self.alpha * td_error 
        
        """
        
    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((81, 3))    
       
