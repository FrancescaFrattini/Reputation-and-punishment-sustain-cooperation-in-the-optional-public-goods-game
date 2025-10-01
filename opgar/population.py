import copy
import json
import logging
from collections import Counter, defaultdict
from itertools import product
import os
from time import time
import numpy as np
import pandas as pd
from tqdm import trange

from .norm import _Norm
from .strategy import _Strategy
from .agent import _Agent, QLearningAgent
from .utils import deprecated, Utils
from .strategy import _Strategy

class Population:
    """Simulate a population of agents playing public goods games.

    Generate a population of agents to play the Optional Public Good Game (OPGG).

    Examples:
        Define a Configuration object containing all the parameters for a single simulation. Create a Population object
        using the parameter dictionary. Run the simulation. Results will be automatically exported.

        >>> import opgar
        >>> C = Configuration(...) # See Configuration class docstring.
        >>> P = Population(C)
        >>> P.simulate()
    """

    def __init__(self, config):
        """
        Create a Population object ready to start a simulation.

        Args:
            config (opgar.Configuration): Object of simulation parameters.
        """
        # Parameters
        self.config = config
        self.strategies = set(config.composition.keys())

        self.actions = [1, 0, None]

        # Generate agents with strategy distribution
        self.agents = self._generate_population(config.N, config.composition)
        self.agents_by_strategy = self._generate_population_by_strategy(
            self.agents, self.strategies
        )
        self.social_norm_type = config.social_norm
        self.social_norm = _Norm(config.social_norm)

        # Granular record of actions
        self.track_strategy_actions = True

    def simulate(self, t_step=None, job_id="", rng_seed=None, disable_bar=False, disable_export=False, transition_matrix_batch=1, record_actions_by_strategy=True):
        """
        Simulate multiple rounds of public goods games

        Args:
            t_step (int): Export data every t_step periods. For efficient memory usage in very long simulations. Default is None.
            job_id (str): Prepend all exported data files with this job_id.
            disable_bar (bool): Remove the tqdm progress bar if True. Default is False.
            disable_export (bool): Do not export datafiles if True. Default is False.

        Returns:
            Tuple of dataframes (avg_payoffs, composition, transitions, punishment_tracker, action_tracker, reputation_tracker)

        Notes:
            Use t_step in conjunction with disable_export=False for long simulations to reduce load on system memory. Every t_step time-steps a new dataframe will be exported. 
            
        """

        logging.info("New Simulation starting")

        if record_actions_by_strategy:
            self.track_strategy_actions = True

        using_batch = True if self.config.t > 250000 else False
        t_start = 0
        t_end = self.config.t
        if t_step is None:
            t_step = self.config.t
        batches = list(range(t_start, t_end, t_step)) + [t_end]

        self.groups_of_players_IDs = self._get_groups()

        for (batch_start, batch_end) in zip(batches[:-1], batches[1:]):
            period_results = {}.fromkeys(range(batch_start, batch_end))
            punishment_tracker = {}.fromkeys(range(batch_start, batch_end))
            cooperative_action_tracker = [None] * (batch_end - batch_start)
            reputation_tracker = [None] * (batch_end - batch_start)
            strategy_actions_tracker = {}.fromkeys(range(batch_start, batch_end))
            transition_matrix = self._generate_transition_matrix(self.actions, 
                                                                 int((batch_end - batch_start + 1) / transition_matrix_batch))

            for t in trange(batch_start, batch_end, desc=f"T=[{batch_start:,}-{batch_end:,}]", disable=disable_bar):
                logging.info(f"T={t} starting")

                # group mixing at each timestep
                if np.random.random() < self.config.delta:
                    self.groups_of_players_IDs = self._get_groups()

                all_action_tracker = {}.fromkeys(["_" .join(s) for s in product(self.strategies, ["1", "0", "None"])], 0)

                # First game
                self._play_public_good_game(all_action_tracker)
                if self.social_norm_type:
                    self._update_reputations()
                punishment_tracker[t] = self._punish(punishment_tracker[t])
                
                # Following games occur with probability omega
                for _ in range(0, self.config.omega - 1):
                    self._play_public_good_game(all_action_tracker)
                    if self.social_norm_type:
                        self._update_reputations()
                    punishment_tracker[t] = self._punish(punishment_tracker[t])

                if self.config.exploration_rate > self.config.minimum_exploration_rate:
                    self.config.exploration_rate *= 0.9995

                # Neaten results
                period_results[t] = self._get_period_result(transition_matrix[int(t / transition_matrix_batch)])
                punishment_tracker[t] = self._neaten_punishment_results(punishment_tracker[t])
                strategy_actions_tracker[t] = all_action_tracker

                """
                # Evolution
                if use_group_selection:
                    self._evolve_group_selection(groups_of_players_IDs, transition_matrix)
                    self._mutate()
                """
                
                # Gather extra information and reset
                period_results[t]["Fitness"] = self._get_population_fitness(period_results[t])
                cooperative_action_tracker[t % t_step] = self._record_cooperative_actions()
                reputation_tracker[t % t_step] = self._record_reputations()
                self._reset_population()

            # ----------------------------------------------------------------------
            # POST-PROCESSING OF EACH BATCH
            # ----------------------------------------------------------------------
            processing_start = time()

            os.makedirs(os.path.dirname("csv/"), exist_ok=True)
            os.makedirs(os.path.dirname("json/"), exist_ok=True)
            os.makedirs(os.path.dirname("csv/transitions/"), exist_ok=True)

            # Average Payoffs
            #avg_payoffs = pd.DataFrame([period_results[t]["Average Payoffs"] for t in range(batch_start, batch_end)])
            avg_payoffs = pd.concat([period_results[t]["Average Payoffs"] for t in range(batch_start, batch_end)], axis=1).transpose()
            avg_payoffs.index = np.arange(batch_start, batch_end)
            avg_payoffs = avg_payoffs.astype("float16")

            # Population State
            composition = pd.concat([period_results[t]["Composition"] for t in range(batch_start, batch_end)], axis=1).transpose()
            composition.index = np.arange(batch_start, batch_end)
            composition = composition.astype("float16")

            # Actions
            action_tracker = pd.DataFrame(cooperative_action_tracker, columns=["Cooperative", "Non-Cooperative", "Loner"], index=range(batch_start, batch_end))
            action_tracker = action_tracker.astype("float16")
            strategy_actions_tracker = pd.DataFrame.from_dict(strategy_actions_tracker, orient="index", dtype="int16")

            transitions = []
            # Actions transitions
            for matrix in transition_matrix:
                transition = [(str(outerKey), str(innerKey), innerVal) for outerKey, outerVal in matrix.items() for innerKey, innerVal in outerVal.items() if innerVal != 0]
                transitions.append(pd.DataFrame(transition, columns=["Source", "Destination", "#"]))
            # Reputations
            reputation_tracker = pd.DataFrame(reputation_tracker, columns=["Good", "Medium", "Bad"], index=range(batch_start, batch_end))
            reputation_tracker.astype("float16")
            # Actions transitions
            actions_transitions = pd.concat([pd.Series(period_results[t]["Transitions"]) for t in range(batch_start, batch_end)], axis=1).transpose()
            actions_transitions.index = np.arange(batch_start, batch_end)

            all_rankings = sorted({r for t in range(batch_start, batch_end) for r in period_results[t]["Q values"].index})
            series_list = [period_results[t]["Q values"].reindex(all_rankings, fill_value=0) for t in range(batch_start, batch_end)]
            q_values_ranking = pd.DataFrame(series_list)

            # Punishments
            punishment_tracker = pd.DataFrame.from_dict(punishment_tracker, orient="index", dtype="float16")
            if not disable_export:
                punishment_tracker.to_csv(f"csv/j{job_id}_punishments_{batch_start}.csv")

                if t_step is not None:
                    batch_code = f"_{batch_start}"
                else:
                    batch_code = ""
                action_tracker.to_csv(f"csv/j{job_id}_actions{batch_code}.csv")
                avg_payoffs.to_csv(f"csv/j{job_id}_payoffs{batch_code}.csv")
                composition.to_csv(f"csv/j{job_id}_composition{batch_code}.csv")
                reputation_tracker.to_csv(f"csv/j{job_id}_reputations{batch_code}.csv")
                for batch_num, transition in enumerate(transitions):
                    transition.to_csv(f"csv/transitions/j{job_id}_transitions{batch_code}_{batch_num}.csv")
                q_values_ranking.to_csv(f"csv/j{job_id}_q_values_rankings_{batch_start}.csv")
                actions_transitions.to_csv(f"csv/j{job_id}_transitions_per_timestep_{batch_start}.csv", index=True)


                if self.track_strategy_actions:
                    strategy_actions_tracker.to_csv(f"csv/j{job_id}_granular_actions{batch_code}.csv")

            processing_end = time()
            logging.info(f"---> Export Batch Data ---> {processing_start-processing_end} seconds elapsed")

        if not disable_export:
            with open(f"json/j{job_id}_config.json", "w") as f:
                json.dump(self.config.to_dict(rng_seed), f)

        if batch_end == self.config.t or disable_export is True:
            return avg_payoffs, composition, transitions, punishment_tracker, action_tracker, reputation_tracker
        else:
            logging.warning("Batch processing used. Results are automatically exported as .csv/.json files.")
            return

    def _get_groups(self, N=None, n=None):
        """
        Return a list of lists representing the player IDs for each group

        Args:
            N (int, optional): Size of the Population. Defaults to None.
            n (int, optional): Size of a single group. Defaults to None.

        Returns:
            groups_of_players_IDs (list): List of agent ID numbers (ints).
        """
        if N is None or n is None:
            N = self.config.N
            n = self.config.n

        player_IDs = np.arange(N)
        np.random.shuffle(player_IDs)
        groups_of_players_IDs = np.reshape(player_IDs, (int(N / n), n)) 
        return groups_of_players_IDs

    def _record_reputations(self):
        """
        Return a tuple of the proportion of good, OK, bad people. 
        """
        reps = Counter([agent.reputation for agent in self.agents])
        return (reps[rep]/self.config.N for rep in [1, 0, -1])

    def _record_cooperative_actions(self):
        """
        Return the proportion of cooperative, defective, loner actions at a single time-step.
        """
        actions_C, actions_D, actions_L = 0, 0, 0
        N = self.config.N

        for agent in self.agents:
            if agent.tracker[-1] == 1:
                actions_C += 1
            elif agent.tracker[-1] == 0:
                actions_D += 1
            else:
                actions_L += 1
        return (actions_C/N, actions_D/N, actions_L/N)

    def _get_population_fitness(self, period_results):
        """Calculate the fitness of the population where F = x1*payoff(x1) + x2*payoff(x2) + ... + xn*payoff(xn).

        Args:
            period_results (pandas.Series): Series containing summary statistics about a single time-step of the simulation.

        Returns:
            fitness (float): The population fitness at a single time-step.
        """ 
        #average payoff for each strategy
        average_payoffs = period_results["Average Payoffs"]
        #proportion of each strategy
        composition = period_results["Composition"]
        #weighted average
        fitness = (average_payoffs * composition).sum()
        return fitness

    def _punish(self, temp_tracker):
        """
        Each player in each group has the opportunity to punish any and all of the other players in his group.

        Args:
            temp_tracker (dict): Temporary container of all the results from the present time-step.
        """

        if "nopunish" in self.config._meta_data["strategy group"] or \
            "Q-Learning no punishment" in self.config._meta_data["strategy group"]:
            return

        if temp_tracker is None:
            temp_tracker = {(src, dst): 0 for src, dst in product([1, 0, None],[1, 0, None])}

        # logging.info("Punishment period starting")
        total_punishments = 0
        for group in self.groups_of_players_IDs:
            for playerID in group:
                # Each person in the group punishes the others 
                punishing_agent = self.agents[playerID]

                # Players "to be or not to be" punished
                rest_of_the_group = [self.agents[ID] for ID in group if ID != playerID]

                for recipient in rest_of_the_group:
                    # Decide whether opponent needs punishment
                    opponents_last_action = recipient.tracker[-1]
                    punishment_needed = punishing_agent.choose_punishment(opponents_last_action)

                    # Apply punishment, agent pays the punishment cost, opponent pays the punishment penalty
                    if punishment_needed:
                        punishing_agent.utility -= self.config.gamma
                        recipient.utility -= self.config.beta

                        temp_tracker[(punishing_agent.tracker[-1], recipient.tracker[-1])] += 1
                        total_punishments += 1
                        """
                        logging.debug(
                            f"A{punishing_agent.ID}(s={punishing_agent.strategy['ID']}) punished "
                            f"A{recipient.ID}(s={recipient.strategy['ID']}) "
                        )

        logging.info(
            f"Punishment period ending: {total_punishments} punishment(s) enacted"
        )
        """

        return temp_tracker

    def _neaten_punishment_results(self, punishment_tracker):
        """
        Convert the punishment result dictionary from counts to a proportion of total number of punishments.

        Args:
            punishment_tracker (dict): Dictionary tracking the source and target of punishment. i.e. a Cooperator punished a defector exactly 7 times, and punished a loner 2 times.
        
        Return:
            punishment_tracker (dict): Updated tracker.
        """

        # Model does not use punishment
        if punishment_tracker is None:
            return

        total_punishments = sum(punishment_tracker.values())
        if total_punishments != 0:
            for direction, punishment_amount in punishment_tracker.items():
                punishment_tracker[direction] = punishment_amount / total_punishments
        return punishment_tracker

    def _update_reputations(self):
        """
        Iterate through the population and update their reputations based on the social norm and their previous action.
        """

        # If we don't care about reputation
        if self.social_norm is None:
            return
        """
        logging.info(f"Reputations are updated ('{self.social_norm_type}')")
        """
        for agent in self.agents:
            most_recent_action = agent.tracker[-1]
            current_reputation = agent.reputation
            new_reputation = self.social_norm._assign_reputation(most_recent_action)
            agent.reputation = new_reputation
            logging.debug(
                f"A{agent.ID}(old={current_reputation}, action={most_recent_action}, new={new_reputation})"
            )

    def _play_public_good_game(self, strategy_action_tracker):
        """
        Simulate the public good game (PGG) once for each player in the population

        Args:
            groups_of_players_IDs (list): List of lists describing the groups in which the OPGG is to be played.
            strategy_action_tracker (dictionary): Temporary dictionary tracking the number of C/D/L actions within this time-step.
        """
        """
        # Setup time-step results
        logging.info("Public Goods Game beginning")
        """

        # ----------------------------------------------------------------------
        # Play PGG
        # ----------------------------------------------------------------------
        
        # group size
        n = len(self.groups_of_players_IDs[0])

        # Play PGG in groups of n
        for group in self.groups_of_players_IDs:
            # Each player calculates the average reputation of the group based on the *rest* of the group
            avg_reps = {}
            for playerID in group:
                avg_reps[playerID] = sum([self.agents[ID].reputation for ID in group if ID != playerID]) / (n - 1)

            # Players decide to contribute 1, contribute 0, or not participate (None)
            group_contribution = [
                self.agents[ID]._choose_action(average_reputation=avg_reps[ID], epsilon=self.config.exploration_rate)
                for ID in group
            ]
            # Possible cases
            #   1. (n-1) Loners -> SKIP PGG -> EVERYONE gets sigma as reward
            #   2. Any non-zero amount of Cooperators/Defectors plays PGG as normal

            group_actions = Counter(group_contribution)
            if group_actions[None] >= n - 1:
                # OPGG skipped -> everyone gets the loner's payoff (sigma)
                for playerID, contribution in zip(group, group_contribution):
                    self.agents[playerID].tracker.append(None)
                    self.agents[playerID].utility += self.config.sigma

                logging.debug(
                    f"Group of agents ({group}) did not play the PGG, everyone receives {self.config.sigma}."
                )
                
                if self.track_strategy_actions:
                    for playerID in group:
                        strategy_action_tracker[self.agents[playerID].strategy["ID"]+"_"+"None"] += 1
            else:
                # Normal PGG, players' reward is calculated as (# of contributors * r / # of players in the group (except loners))
                try:
                    total_contribution = group_actions[1]  # number of contributors
                except KeyError:
                    total_contribution = 0

                total_participating = (
                    group_actions[1] + group_actions[0]
                )  # number of defectors
                payoff_per_player = (
                    total_contribution * self.config.r / total_participating
                ) 
                #variable for Q-Learning agents
                old_utility = self.agents[playerID].utility 
                for playerID, contribution in zip(group, group_contribution):
                    self.agents[playerID].tracker.append(contribution)
                    logging.debug("Player %s chose action %s", playerID, self.agents[playerID].tracker[-1])
                    if contribution == 1: #payoff for cooperators
                        self.agents[playerID].utility -= 1 # contribution given by the player
                        self.agents[playerID].utility += payoff_per_player
                    elif contribution == 0: #payoff for defectors
                        self.agents[playerID].utility += payoff_per_player
                    else: #Loner
                        self.agents[playerID].utility += self.config.sigma
                    # Q-Learning agent learns
                    if self.agents[playerID].strategy["behavioural"] == "XII":
                        counts = group_actions.copy()
                        counts[contribution] -= 1
                        reward = self.agents[playerID].utility - old_utility
                        self.agents[playerID].learn(contributions=counts)
                        
                if self.track_strategy_actions:
                    for playerID, contribution in zip(group, group_contribution):    
                        strategy_action_tracker[self.agents[playerID].strategy["ID"]+"_"+str(contribution)] += 1

                logging.debug(
                    f"Group of agents ({group}) played the PGG, average payoff was {round(payoff_per_player, 2)} each "
                    f"to {total_participating} agents"
                )


    def _get_period_result(self, transitions):
        """
        Get period results in a pandas Series.

        Returns:
            pandas.Series: Contains all information regarding the present time-step.
        """

        # Save all results for the time-step here (possibly multiple rounds of games)
        period_result = {
            "Payoffs": defaultdict(float),
            "Composition Count": defaultdict(int),
            "Composition": defaultdict(float),
            "Actions per strategy": defaultdict(int),
            "Transitions": {a: {b: 0 for b in self.actions if b != a} for a in self.actions},
            "Q values": Counter()
            }
        
        # Record total strategy payoffs, strategy composition
        for agent in self.agents:
            period_result["Payoffs"][agent.strategy["ID"]+"_"+str(agent.tracker[-1])] += (agent.utility - 1)
            period_result["Composition Count"][agent.strategy["ID"]] += 1
            period_result["Actions per strategy"][agent.strategy["ID"]+"_"+str(agent.tracker[-1])] += 1
            ranking_q_values = tuple(int(x) for x in np.argsort(agent.getLastRow())[::-1])
            period_result["Q values"][ranking_q_values] += 1

            # actions transition tracker
            if len(agent.tracker) >= 2:
                transitions[agent.tracker[-2]][agent.tracker[-1]] += 1
                if agent.tracker[-1] != agent.tracker[-2]:
                    period_result["Transitions"][agent.tracker[-2]][agent.tracker[-1]] += 1
                    
        for strategy in self.strategies:
            # Get population composition as a proportion instead of relative size
            period_result["Composition"][strategy] = (
                period_result["Composition Count"][strategy] / self.config.N
            )
        # Average strategy payoffs by number of agents using the strategy and by action
        period_result["Average Payoffs"] = {
            k: period_result["Payoffs"][k] / period_result["Actions per strategy"][k]
            for k in period_result["Payoffs"]}
    
        # Flatten period_result from dict-of-dicts to single-level dict
        #   e.g. {"a1": {"a2": []}, "b1"={"b2":[]}} -> {('a1', 'a2'): [], ('b1', 'b2'): []}
        #   This makes it easier to convert it to a MultiIndex Series/DataFrame later when concatenating results over
        #   multiple time-steps.
        flat_dict = Utils.flatten_nested_dict(period_result)

        classed_series = pd.Series(flat_dict)
        classed_series.index.names = ["Statistic", "Strategy"]
        return classed_series


    def _evolve_group_selection(self, transitions):
        """
        Evolution implementation using group-selection. 

        Players encounter another player from their own group with probability 1-m and from another randomly chosen 
        group with probability m. An individual i who encounters an individual j imitates j with probability 
        W_j / (W_j + W_i) where W_x is the payoff of individual x including the costs of giving or receiving punishment.

        Args:
            transitions (dict): Nested dictionary of counts of strategy transitions.
        """

        evolving_agent_id = np.random.choice(range(self.config.N))
        evolving_agent = self.agents[evolving_agent_id]
        if isinstance(evolving_agent, QLearningAgent):       
            return
        evolving_agent_strategy = evolving_agent.strategy["ID"]
        evolving_agent_group = [group for group in self.groups_of_players_IDs if evolving_agent_id in group][0]
                
        if np.random.random() < 1 - self.config.m:
            # selection from another individual in the same group
            other_agent_id = np.random.choice(evolving_agent_group)
            while other_agent_id == evolving_agent_id:
                other_agent_id = np.random.choice(evolving_agent_group)
            other_agent = self.agents[other_agent_id]
            other_agent_strategy = other_agent.strategy["ID"]
        else:
            # selection from someone in another group
            other_group = self.groups_of_players_IDs[np.random.randint(len(self.groups_of_players_IDs))]
            while evolving_agent_id in other_group:
                other_group = self.groups_of_players_IDs[np.random.randint(len(self.groups_of_players_IDs))]
            other_agent_id = np.random.choice(other_group)
            other_agent = self.agents[other_agent_id]
            other_agent_strategy = other_agent.strategy["ID"]
        
        probability_of_update = np.exp(other_agent.utility) / (np.exp(evolving_agent.utility) + np.exp(other_agent.utility))
        if probability_of_update < 0: 
            logging.warn(f"Probability of update ('{probability_of_update}') is negative!")

        if np.random.random() < probability_of_update \
            and other_agent.strategy["behavioural"] not in _Strategy.learner_strategies:
            self._change_agent_strategy(evolving_agent, other_agent_strategy)
            if evolving_agent_strategy != other_agent_strategy:
                transitions[evolving_agent_strategy][other_agent_strategy] += 1
        
    def _mutate(self):
        """
        With probability epsilon, introduce a mutant into the population.
        If the chosen mutant is a Q-Learning agent, the method does nothing
    
        """
        if np.random.random() < self.config.epsilon:
            mutant = np.random.choice(self.agents)
        
            if isinstance(mutant, QLearningAgent):
                return
            pool = [
                s for s in self.strategies
                if s.split("_")[0] in _Strategy.standard_strategies and s != mutant.strategy["ID"]
            ]
            if not pool:                     
                return
            mutated_strategy = np.random.choice(pool)
            self._change_agent_strategy(mutant, mutated_strategy)

    def _change_agent_strategy(self, agent, new_strategy):
        """
        Change an agent's strategy updating the Population.

        Args:
            agent (opgar.Agent): An agent to mutate
            new_strategy (str): A new strategy.
        """
        root = new_strategy.split("_")[0]

        if not isinstance(agent, QLearningAgent) and root in _Strategy.learner_strategies:
            logging.warning(
                f"Trying to assign '{new_strategy}' to a Q-Learning agent. Ignoring")
            return
        
        # Update strategy in Agent object
        old_strategy = agent.strategy["ID"]
        agent.strategy = {
            key: val
            for key, val in zip(["behavioural", "punishment"], new_strategy.split("_"))
        }
        agent.strategy["ID"] = new_strategy

        # Update population census by strategy
        if new_strategy not in self.agents_by_strategy.keys():
            self.agents_by_strategy[new_strategy] = set()
        self.agents_by_strategy[old_strategy].remove(agent)
        self.agents_by_strategy[new_strategy].add(agent)
        
        logging.info(
            f"Agent {agent.ID} switched from {old_strategy} to {new_strategy}."
        )
        

    def _reset_population(self):
        """
        Reset all agents and the population between time-steps.

        Things that are reset:
            An agent's utility is reset to 1

        Things that do not reset:
            An agent's reputation
            Any population parameters
        """

        # logging.info("Agents reset")
        for agent in self.agents:
            agent.utility = 1
            # agent.reputation = 1

    @staticmethod
    def _log_series(name, series):
        """
        Given a pandas.Series, return a one-line string representation for logging.
        """
        return f"{name}: {', '.join([k[1] + '->' + str(round(v, 4)) for k, v in series.items()])}"

    def _generate_population(self, N, composition):
        """
        Create the population of agents with the required distribution of strategies.

        Given some proportion of strategies within the population as a dictionary of strategy: 'proportion of 1' pairs,
        convert the proportions to numbers of agents by multiplying by the size of the population. These may or may not
        be non-integers, so we use the Largest Remainder method to assign the last few agents.

        Args:
            N (int): Size of the network
            composition (dict): Dictionary of strategies and their proportions within the population

        Returns:
            List of agents with a strategy distribution equal to that of composition
        """

        # Partition N players into fractions of 1 as accurately as possible
        proportions = Population._distribute_over_N(composition, N)

        agents = []
        id_counter = 0
        for strategy, count in proportions.items():
            for _ in range(int(count)):
                if strategy.split("_")[0] == "XII":
                    # If the strategy is QLearning, create a QLearningAgent
                    agents.append(QLearningAgent(ID=id_counter, strategy=strategy,  
                                                alpha=self.config.alpha, discount_factor=self.config.discount_factor))
                else:
                    agents.append(_Agent(ID=id_counter, strategy=strategy))
                id_counter += 1

        return agents


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
        logging.debug(Population._log_series("Allocated: ", temp_dist))

        # How many remaining agents need to be assigned a class
        remaining = N - sum(temp_dist)
        logging.debug(f"Remaining: {remaining}")
        if remaining == 0:
            return temp_dist

        logging.info("Approximating New Population using Largest Remainder Method")
        while remaining > 0:
            # get remainders
            unallocated_dist = dist_over_N - temp_dist
            logging.debug(Population._log_series("Unallocated: ", unallocated_dist))

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

    def __str__(self):
        s = []
        for agent in self.agents:
            s += [
                f"A(ID={agent.ID}, s={agent.strategy['ID']}, u={round(agent.utility, 4)}, r={agent.reputation[-1]})"
            ]
        output = "\n".join(s)
        return output
    