import logging
import random

import numpy as np
from .strategy import _Strategy

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
        self.tracker = None
        logging.info(
            f"Agent {self.ID} created with r={self.reputation} & s={self.strategy}"
        )

    def _choose_action(self, average_reputation):
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
# uses Strategy XII - Q-Learning without states (only 3 actions)

    __slots__ = _Agent.__slots__ + [
    "q_values", "alpha", "gamma", "epsilon"
    ]

    ACTIONS = [1, 0, None]  # Actions: cooperate (1), defect (0), withdraw (None)

    def __init__(self, ID, alpha=0.1, gamma=0.99, epsilon=0.1):
        # NB: strategy root “XII”, no punischment component
        super().__init__(ID, strategy="XII_NNN")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q_values = np.zeros(len(self.ACTIONS))

    def _choose_action(self, average_reputation):
        # ignoring average_reputation for now, as this is a Q-Learning agent
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        return self.ACTIONS[int(np.argmax(self.q_values))]

    def learn(self, reward, action_taken):
        idx = self.ACTIONS.index(action_taken)
        td_error = reward - self.q_values[idx]      # no next‑state
        self.q_values[idx] += self.alpha * td_error
