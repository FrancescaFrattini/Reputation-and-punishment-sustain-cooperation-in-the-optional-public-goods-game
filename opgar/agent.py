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
    """Q-learning agent whose state is a rolling history of group actions."""

    __slots__ = _Agent.__slots__ + [
        "alpha", "discount_factor", "q_table", "current_state", "n",
        "state_to_idx", "idx_to_state", "state_history",
    ]

    ACTIONS = [0, 1, None]  # defect, cooperate, abstain

    def __init__(self, ID, strategy, group_size, alpha=0.1,
                 discount_factor=0.1, n=1):
        super().__init__(ID, strategy=strategy)

        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 10:
            raise ValueError("n must be an integer between 1 and 10")

        self.alpha = alpha
        self.discount_factor = discount_factor
        self.n = n

        # The table starts without states.  Rows are created lazily when a
        # previously unseen observation history is encountered.
        self.q_table = np.zeros((0, len(self.ACTIONS)), dtype=float)
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.state_history = []
        self.current_state = None

    def _choose_action(self, epsilon, average_reputation=None):
        """Choose an action with epsilon-greedy exploration."""
        if (self.current_state is None or random.random() < epsilon
                or self._is_uninitialized()):
            self.tracker.append(random.choice(self.ACTIONS))
        else:
            best_action_index = np.argmax(self.q_table[self.current_state])
            self.tracker.append(self.ACTIONS[best_action_index])
        return self.tracker[-1]

    def learn(self, reward, contributions):
        """Update Q after observing the group's actions for this round.

        ``contributions`` is converted into one action-count triple.  The
        state is the ordered tuple of the most recent triples, up to ``n``.
        Therefore, during the first ``n`` rounds its length grows from 1 to
        ``n``; afterwards it behaves as a circular observation window.
        """
        observed_triple = Utils.from_counter_to_tuple(counter=contributions)
        next_state = self._append_observation(observed_triple)
        next_state_idx = self._get_or_create_state(next_state)

        logging.info(f"agent {self.ID} chose action {self.tracker[-1]} current Q-table is {self.q_table} \
        \n observed_triple {observed_triple} \n next_state {next_state} \n next state id {next_state_idx}")

        # At the first round there is no previous state/action pair to update.
        if self.current_state is not None:
            action_idx = self._action_to_index(self.tracker[-1])
            current_qvalue = self.q_table[self.current_state, action_idx]
            next_qvalue = np.argmax(self.q_table[next_state_idx])
            td_error = reward + self.discount_factor * next_qvalue - current_qvalue
            self.q_table[self.current_state, action_idx] = (
                current_qvalue + self.alpha * td_error
            )

        logging.info(f"updated q-table for agent {self.ID} :\n {self.q_table}")

        self.current_state = next_state_idx

    def _append_observation(self, observed_triple):
        """Add an observation and return the immutable history-state key."""
        self.state_history.append(observed_triple)
        if len(self.state_history) > self.n:
            self.state_history.pop(0)
        return tuple(self.state_history)

    def _get_or_create_state(self, state):
        """Return a state row, adding a zero-initialized row when necessary."""
        state_idx = self.state_to_idx.get(state)
        if state_idx is not None:
            return state_idx

        state_idx = len(self.state_to_idx)
        self.state_to_idx[state] = state_idx
        self.idx_to_state[state_idx] = state
        self.q_table = np.vstack((
            self.q_table,
            np.zeros((1, len(self.ACTIONS)), dtype=float),
        ))
        return state_idx

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
        return action

    def _is_uninitialized(self):
        """Whether the currently observed history has never been updated."""
        return np.all(self.q_table[self.current_state] == 0)
