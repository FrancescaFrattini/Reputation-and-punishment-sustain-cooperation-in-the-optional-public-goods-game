import copy
import logging
import numpy as np

class Generator:
    @staticmethod
    def _log_series(name, series):
        """
        Given a pandas.Series, return a one-line string representation for logging.
        """
        return f"{name}: {', '.join([k[1] + '->' + str(round(v, 4)) for k, v in series.items()])}"
    
    @staticmethod
    def _generate_population_by_strategy(agentSet, strategies):
        """
        Create an alternative view of Population.agents by strategy.

        Args:
            agentSet (list): List of agent objects

        Returns:
            Dictionary of agents where the keys:value pairs are strategies and lists of the agents running them
        """
        agents_by_strategy = {}.fromkeys(strategies)

        # Using "agents_by_strategy = {}.fromkeys(strategies, set())" makes each value point to the same set object
        for key, value in agents_by_strategy.items():
            agents_by_strategy[key] = set()

        # Organise each agent by their strategy
        for agent in agentSet:
            agents_by_strategy[agent.strategy["ID"]].add(agent)

        return agents_by_strategy

    @staticmethod
    def _generate_transition_matrix(actions, n_batches):
        """Initialise a strategy transition matrix.

        Args:
            actions (list): List of available actions

        Returns:
            matrix (dict): Nested dicts in the form of an KxK matrix where K is the number of actions.
        """
        matrix = []
        for _ in range (n_batches):
            template = {}.fromkeys(actions, 0)
            mat = {}.fromkeys(actions)
            for action in actions:
                mat[action] = copy.deepcopy(template)
            matrix.append(mat)

        return matrix

    @staticmethod
    def _distribute_over_N(dist, N):
        """
        Distribute N into k classes as specified by the distribution in dist.
        
        Args:
            dist (list): The distribution of the population over the strategies.
            N (int): The total number of players to assign to a strategy.
        """

        # Roughly distribute N agents to the k classes by the proportions in dist allowing only integers
        dist_over_N = dist * N
        # dist_over_N = dist_over_N
        temp_dist = dist_over_N.astype(int)
        logging.debug(Generator._log_series("Allocated: ", temp_dist))

        # How many remaining agents need to be assigned a class
        remaining = N - sum(temp_dist)
        logging.debug(f"Remaining: {remaining}")
        if remaining == 0:
            return temp_dist

        logging.info("Approximating New Population using Largest Remainder Method")
        while remaining > 0:
            # get remainders
            unallocated_dist = dist_over_N - temp_dist
            logging.debug(Generator._log_series("Unallocated: ", unallocated_dist))

            # get largest remainder
            max_remainder = max(unallocated_dist)
            logging.debug(f"Max Remainder: {max_remainder}")

            # get list of indices with the max remainder
            idxs = unallocated_dist[unallocated_dist == max_remainder]
            logging.debug(f"Indices: {idxs.index}")

            # Choose strategy to increment (if multiple choose randomly)
            new_strat = np.random.choice(idxs.index)
            logging.debug(f"New Strategy: {new_strat}")

            # Allocate agent
            temp_dist[new_strat] += 1
            remaining -= 1
        return temp_dist

    @staticmethod
    def _log_series(name, series):
        """
        Given a pandas.Series, return a one-line string representation for logging.
        """
        return f"{name}: {', '.join([k[1] + '->' + str(round(v, 4)) for k, v in series.items()])}"    