import copy
import json
import logging
from collections import Counter, defaultdict
from itertools import product
import os
import random
from time import time
import numpy as np
import pandas as pd
from tqdm import trange

from .norm import _Norm
from .strategy import _Strategy
from .agent import _Agent, QLearningAgent
from .utils import deprecated, Utils
from .generator import _Generator

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
        self.agents_by_strategy = _Generator._generate_population_by_strategy(
            self.agents, self.strategies
        )
        self.social_norm_type = config.social_norm
        self.social_norm = _Norm(config.social_norm)

        # Granular record of actions
        self.track_strategy_actions = True
        self.exploration_rate = config.exploration_rate
        self.qtable_changes = defaultdict(int)
        self.subgroups = self._build_subgroups()

    def simulate(self, t_step=None, job_id="", rng_seed=None, disable_bar=False, disable_export=False, transition_matrix_batch=None, record_actions_by_strategy=True, run_id=""):
        """
        Simulate multiple rounds of public goods games

        Args:
            t_step (int): Export data every t_step periods. For efficient memory usage in very long simulations. Default is None.
            job_id (str): Prepend all exported data files with this job_id.
            run_id (str): Prepend all exported data files with this run_id.
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

        #self.groups_of_players_IDs = self._get_groups()
        self.groups_of_players_IDs = self._get_groups_inside_subgroups()

        if transition_matrix_batch is None:
            transition_matrix_batch = self.config.t

        for (batch_start, batch_end) in zip(batches[:-1], batches[1:]):
            period_results = {}.fromkeys(range((batch_end - batch_start) * self.config.omega))
            punishment_tracker = {}.fromkeys(range((batch_end - batch_start) * self.config.omega))
            cooperative_action_tracker = [None] * (batch_end - batch_start)
            reputation_tracker = {}.fromkeys(range((batch_end - batch_start) * self.config.omega))
            strategy_actions_tracker = {}.fromkeys(range((batch_end - batch_start) * self.config.omega))
            transition_matrix = _Generator._generate_transition_matrix(self.actions, 
                                                                 int((batch_end - batch_start + 1) / transition_matrix_batch))

            for t in trange(batch_start, batch_end, desc=f"T=[{batch_start:,}-{batch_end:,}]", disable=disable_bar):
                logging.info(f"T={t} starting")

                batch_len = (batch_end - batch_start) * self.config.omega
                if (t - batch_start) * self.config.omega in {
                    round(batch_len * 0.5)
                }:
                    """        
                    # DEBUG
                    with open(f"subgroups_round_{((t - batch_start) * self.config.omega)}.txt", "w") as f:
                        for strat, members in self.subgroups.items():
                            f.write(strat + "\n")
                            for id in members:
                                f.write(f"{id, self.agents[id].strategy['behavioural']}\n")          
                    """      
                    self.subgroups = self._rotate_subgroups()
                    self.groups_of_players_IDs = self._get_groups_inside_subgroups()
                    self.agent_to_group = {
                        agent_id: group
                        for group, ids in self.q_learner_groups.items()
                        for agent_id in ids
                    }
                else:
                    # group mixing at each timestep
                    if self.config.reset_exploration_rate is None:
                        if np.random.random() < self.config.delta:
                            self.groups_of_players_IDs = self._get_groups_inside_subgroups()
                            self.agent_to_group = {
                                agent_id: group
                                for group, ids in self.q_learner_groups.items()
                                for agent_id in ids
                            }
                

                for n in range(self.config.omega):
                    #all_action_tracker = {}.fromkeys(["_" .join(s) for s in product(self.strategies, ["1", "0", "None"])], 0)
                    all_action_tracker = {}.fromkeys(["_".join(s) for s in product(["XII_VI_NNN", "XII_VII_NNN", 
                                                   "VI_NNN", "VII_NNN"], ["1", "0", "None"])], 0)

                    self._play_public_good_game(all_action_tracker)
                    if self.social_norm_type:
                        self._update_reputations()
                    punishment_tracker[t] = self._punish(punishment_tracker[t])
       
                    if self.exploration_rate > self.config.minimum_exploration_rate:
                        self.exploration_rate *= self.config.epsilon_decay

                    # Neaten results
                    period_results[(t - batch_start) * self.config.omega + n] =  \
                            self._get_period_result(transition_matrix[int(t / transition_matrix_batch)])
                    punishment_tracker[t] = self._neaten_punishment_results(punishment_tracker[t])
                    strategy_actions_tracker[(t - batch_start) * self.config.omega + n] = all_action_tracker

                    """
                    # Evolution
                    if use_group_selection:
                        self._evolve_group_selection(groups_of_players_IDs, transition_matrix)
                        self._mutate()
                    """
                
                    # Gather extra information and reset
                    period_results[(t - batch_start) * self.config.omega + n]["Fitness"] = \
                            self._get_population_fitness(period_results[(t - batch_start) * self.config.omega + n])
                    cooperative_action_tracker[t % t_step] = self._record_cooperative_actions()
                    reputation_tracker[(t - batch_start) * self.config.omega + n] = self._record_reputations()

                    # Reset exploration rate and group mixing
                    if self.config.reset_exploration_rate is not None and \
                            ((t - batch_start) * self.config.omega + n) % self.config.reset_exploration_rate == 0:
                        self.exploration_rate = self.config.exploration_rate
                        self.groups_of_players_IDs = self._get_groups_inside_subgroups()
                        self.agent_to_group = {
                            agent_id: group
                            for group, ids in self.q_learner_groups.items()
                            for agent_id in ids
                        }
                        #self.groups_of_players_IDs = self._get_groups()

                    #population reset at the end of each timestep
                    self._reset_population()

            # ----------------------------------------------------------------------
            # POST-PROCESSING OF EACH BATCH
            # ----------------------------------------------------------------------
            processing_start = time()

            os.makedirs(os.path.dirname(f"csv/{run_id}/"), exist_ok=True)
            os.makedirs(os.path.dirname(f"json/{run_id}/"), exist_ok=True)

            # Average Payoffs & Population State & actions transitions
            avg_payoffs = pd.concat([period_results[n]["Average Payoffs"] for n in range((batch_end - batch_start) * self.config.omega)], axis=1).transpose()
            avg_payoffs.index = np.arange((batch_end - batch_start) * self.config.omega)
            avg_payoffs = avg_payoffs.astype("float16")
            population = pd.concat([period_results[n]["Composition"] for n in range((batch_end - batch_start) * self.config.omega)], axis=1).transpose()
            population.index = np.arange((batch_end - batch_start) * self.config.omega)
            population = population.astype("float16")
            action_transitions = pd.concat([pd.Series(period_results[n]["Transitions"]) for n in range((batch_end - batch_start) * self.config.omega)], axis=1).transpose()
            action_transitions.index = np.arange((batch_end - batch_start) * self.config.omega)

            all_avg = sorted({avg for n in range((batch_end - batch_start) * self.config.omega) for avg in period_results[n]["Q values"].keys()})
            
            q_values_ranking = pd.DataFrame([
                {
                    avg: {
                        i: period_results[n]["Q values"].get(avg, {}).get(i, 0)
                        for i in range(3)
                    } for avg in all_avg
                } for n in range((batch_end - batch_start) * self.config.omega)
            ])

            q_values_ranking = q_values_ranking.map(lambda d: {k: d.get(k, 0) for k in [0, 1, 2]} if isinstance(d, dict) else {0: 0, 1: 0, 2: 0})

            # Actions
            action_tracker = pd.DataFrame(cooperative_action_tracker, columns=["Cooperative", "Non-Cooperative", "Loner"], index=range(batch_start, batch_end))
            action_tracker = action_tracker.astype("float16")
            strategy_tracker = pd.DataFrame.from_dict(strategy_actions_tracker, orient="index", dtype="int16")

            transitions = []
            # Actions transitions
            for matrix in transition_matrix:
                transition = [(str(outerKey), str(innerKey), innerVal) for outerKey, outerVal in matrix.items() 
                              for innerKey, innerVal in outerVal.items() if innerVal != 0]
                transitions.append(pd.DataFrame(transition, columns=["Source", "Destination", "#"]))
             # Reputations
            #reputation = pd.DataFrame([reputation_tracker.get(n, (0, 0, 0)) for n in range((batch_end - batch_start) * self.config.omega)], columns=["good", "ok", "bad"])
            reputation = pd.DataFrame([reputation_tracker[n] for n in range((batch_end - batch_start) * self.config.omega)])
            reputation = reputation.astype("float16")

            # Punishments
            #punishment_tracker = pd.DataFrame.from_dict(punishment_tracker, orient="index", dtype="float16")
            if not disable_export:
                #punishment_tracker.to_csv(f"csv{run_id}/j{job_id}_punishments_{batch_start}.csv")

                if t_step is not None:
                    batch_code = f"_{batch_start}"
                else:
                    batch_code = ""
                action_tracker.to_csv(f"csv{run_id}/j{job_id}_actions{batch_code}.csv")
                avg_payoffs.to_csv(f"csv{run_id}/j{job_id}_payoffs{batch_code}.csv")
                population.to_csv(f"csv{run_id}/j{job_id}_composition{batch_code}.csv")
                reputation.to_csv(f"csv{run_id}/j{job_id}_reputations{batch_code}.csv")
                q_values_ranking.to_csv(f"csv{run_id}/j{job_id}_q_values_rankings_{batch_start}.csv")
                action_transitions.to_csv(f"csv{run_id}/j{job_id}_transitions_per_timestep_{batch_start}.csv", index=True)

                for batch_num, transition in enumerate(transitions):
                    transition.to_csv(f"csv{run_id}/j{job_id}_transitions_{batch_start}_{batch_num}.csv", index=True)

                if self.track_strategy_actions:
                    strategy_tracker.to_csv(f"csv{run_id}/j{job_id}_granular_actions{batch_code}.csv")

            processing_end = time()
            logging.info(f"---> Export Batch Data ---> {processing_end-processing_start} seconds elapsed")

        if not disable_export:
            with open(f"json{run_id}/j{job_id}_config.json", "w") as f:
                json.dump(self.config.to_dict(rng_seed), f)

        with open(f"json{run_id}/j{job_id}_table_changes.json", "w") as f:
            json.dump([{str(k): self.qtable_changes[k]} for k in sorted(self.qtable_changes.keys(), key=float)],
                      f, indent=2)

        if batch_end == self.config.t or disable_export is True:
            return avg_payoffs, population, transitions, punishment_tracker, action_tracker, reputation_tracker
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
    
    def _get_groups_inside_subgroups(self, n=None):
        """
        Distributes subgroups of players in groups of size n
        Args:
            n (int, optional): Size of a single group. Defaults to None.
        Returns:
            list of lists representing players IDs for each group, within each subgroup

        """

        if n is None:
            n = self.config.n 

        final_groups = {} 

        for strat_name, agent_ids in self.subgroups.items():
            agent_ids = np.array(agent_ids) 
            np.random.shuffle(agent_ids)   
            groups = np.reshape(agent_ids, (int(len(agent_ids) / n), n))
            final_groups[strat_name] = groups.tolist()

        return sum(final_groups.values(), [])  # flatten the list of lists

    def _rotate_subgroups(self):
        """
        Rotates agents' subgroups
        Q-Learner agents rotate this way:
        IV_NNN -> V_NNN -> VI_NNN -> VII_NNN
        """
        #strat_defectnotbad = [agent.ID for agent in self.agents_by_strategy.get("IV_NNN", [])]
        #strat_defectgood = [agent.ID for agent in self.agents_by_strategy.get("V_NNN", [])]
        strat_lonernotbad = [agent.ID for agent in self.agents_by_strategy.get("VI_NNN", [])]
        strat_lonergood = [agent.ID for agent in self.agents_by_strategy.get("VII_NNN", [])]
        old_groups = self.q_learner_groups.copy() 
        self.q_learner_groups.clear()
        #self.q_learner_groups["IV_NNN"] = old_groups["V_NNN"]
        #self.q_learner_groups["V_NNN"] = old_groups["IV_NNN"]  
        self.q_learner_groups["VI_NNN"] = old_groups["VII_NNN"]   
        self.q_learner_groups["VII_NNN"] = old_groups["VI_NNN"]

        subgroups = {
            #"defectnotbad": strat_defectnotbad + self.q_learner_groups["IV_NNN"],
            #"defectgood": strat_defectgood + self.q_learner_groups["V_NNN"],
            "lonernotbad": strat_lonernotbad + self.q_learner_groups["VI_NNN"],
            "lonergood": strat_lonergood + self.q_learner_groups["VII_NNN"]
        }
        
        return subgroups

    def _record_reputations(self):
        """
        Return a tuple of the proportion of good, OK, bad people. 
        """
        return Counter([
        (
            agent.strategy["behavioural"] + "_" +
            self.agent_to_group.get(agent.ID, "UNKNOWN") + "_" + str(agent.reputation)
        )
            if agent.strategy["behavioural"] == "XII"
            else agent.strategy["ID"] + "_" + str(agent.reputation)
            for agent in self.agents
        ])
        # reps = Counter([agent.reputation for agent in self.agents])
        #return (reps[rep]/self.config.N for rep in [1, 0, -1])

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
            reputations = {}
            for playerID in group:
                reputations[playerID] = sum([self.agents[ID].reputation for ID in group if ID != playerID]) / (n - 1)

            avg_reps = sum(reputations.values()) / len(reputations)
            # Players decide to contribute 1, contribute 0, or not participate (None)
            group_contribution = [
                self.agents[ID]._choose_action(average_reputation=reputations[ID], epsilon=self.exploration_rate)
                for ID in group
            ]
            # Possible cases
            #   1. (n-1) Loners -> SKIP PGG -> EVERYONE gets sigma as reward
            #   2. Any non-zero amount of Cooperators/Defectors plays PGG as normal

            group_actions = Counter(group_contribution)

            if group_actions[None] >= n - 1:
                logging.debug(f"Group of agents ({group}) did not play the PGG, everyone receives {self.config.sigma}.")
                # OPGG skipped -> everyone gets the loner's payoff (sigma)
                for playerID, contribution in zip(group, group_contribution):
                    #self.agents[playerID].tracker.append(contribution)
                    self.agents[playerID].utility += self.config.sigma
                    if self.agents[playerID].strategy["behavioural"] == "XII":
                        self.agents[playerID].learn(reward=self.config.sigma, avg = avg_reps)
                        self.qtable_changes[avg_reps] += 1
                        strategy_action_tracker[self.agents[playerID].strategy["behavioural"] + "_" + next
                            (k for k, v in self.q_learner_groups.items() if self.agents[playerID].ID in v) + "_" + 
                            str(contribution)] += 1
                    else:
                        strategy_action_tracker[self.agents[playerID].strategy["ID"]+"_"+str(contribution)] += 1
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
                avg_payoff = 0
                for playerID, contribution in zip(group, group_contribution):
                    if contribution == None:
                        avg_payoff += 1
                    else:
                        avg_payoff += payoff_per_player
                avg_payoff = round(avg_payoff / self.config.n, 2)
                for playerID, contribution in zip(group, group_contribution):
                    #variable for Q-Learning agents
                    old_utility = self.agents[playerID].utility 
                    #self.agents[playerID].tracker.append(contribution)
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
                        reward = self.agents[playerID].utility - old_utility
                        self.agents[playerID].learn(reward = reward, avg = avg_reps)
                        self.qtable_changes[avg_reps] += 1
                        #strategy_action_tracker[self.agents[playerID].strategy["ID"]+"_"+str(contribution)] += 1
                        strategy_action_tracker[self.agents[playerID].strategy["behavioural"] + "_" + next
                            (k for k, v in self.q_learner_groups.items() if self.agents[playerID].ID in v) + "_" + 
                            str(contribution)] += 1
                    else:
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
            "Q values": defaultdict(Counter)
            }
        
        # Record total strategy payoffs, strategy composition
        for agent in self.agents:
            period_result["Composition Count"][agent.strategy["behavioural"]] += 1
                
            if agent.strategy["behavioural"] == "XII":
                for avg_payoff, q_values in agent.q_table.items():
                    ranking = np.argmax([deq[-1] for deq in q_values])
                    period_result["Q values"][avg_payoff][ranking] += 1
                period_result["Payoffs"][agent.strategy["behavioural"] + "_" + self.agent_to_group.get(agent.ID, "UNKNOWN") + "_" + 
                                    str(agent.tracker[-1])] += (agent.utility - 1)
                period_result["Actions per strategy"][agent.strategy["behavioural"] + "_" + self.agent_to_group.get(agent.ID, "UNKNOWN") + 
                                "_" + str(agent.tracker[-1])] += 1
            #else:
            period_result["Payoffs"][agent.strategy["behavioural"]+"_"+str(agent.tracker[-1])] += (agent.utility - 1)
            period_result["Actions per strategy"][agent.strategy["behavioural"]+"_"+str(agent.tracker[-1])] += 1


            # actions transition tracker
            if len(agent.tracker) >= 2:
                transitions[agent.tracker[-2]][agent.tracker[-1]] += 1
                if agent.tracker[-1] != agent.tracker[-2]:
                    period_result["Transitions"][agent.tracker[-2]][agent.tracker[-1]] += 1
                    
        for strategy in self.strategies:
            strategy_name = strategy.split("_", 1)[0]
            # Get population composition as a proportion instead of relative size
            period_result["Composition"][strategy_name] = (
                period_result["Composition Count"][strategy_name] / self.config.N
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
        proportions = _Generator._distribute_over_N(composition, N)

        agents = []
        id_counter = 0
        for strategy, count in proportions.items():
            for _ in range(int(count)):
                if strategy.split("_")[0] == "XII":
                    # If the strategy is QLearning, create a QLearningAgent
                    agents.append(QLearningAgent(ID=id_counter, strategy=strategy,  
                                                alpha=self.config.alpha, discount_factor=self.config.discount_factor, 
                                                group_size=self.config.n, n = self.config.observation_space))
                else:
                    agents.append(_Agent(ID=id_counter, strategy=strategy))
                id_counter += 1

        return agents    

    def _build_subgroups(self):
        """
        generate subgroups of agents based on their strategy, to be used for group mixing and subgroup rotation.

        Returns:
            subgroups (dict): A dictionary of strategy: list of agent IDs pairs, where each strategy is a key and the value is a list of agent IDs that have that strategy. 
        """

        #strat_defectnotbad = [agent.ID for agent in self.agents_by_strategy.get("IV_NNN", [])]
        #strat_defectgood = [agent.ID for agent in self.agents_by_strategy.get("V_NNN", [])]
        strat_lonernotbad = [agent.ID for agent in self.agents_by_strategy.get("VI_NNN", [])]
        strat_lonergood = [agent.ID for agent in self.agents_by_strategy.get("VII_NNN", [])]
        q_learner = [agent.ID for agent in self.agents_by_strategy.get("XII_NNN", [])]
        np.random.shuffle(q_learner)

        total_q_learner = len(q_learner)
        q_learner_per_group = total_q_learner // 2
        leftovers = total_q_learner % 2

        sizes = [q_learner_per_group] * 2
        for i in range(leftovers):
            sizes[i] += 1

        self.q_learner_groups = {"VI_NNN": [], "VII_NNN": []}#, "VI_NNN": [], "VII_NNN": [] }

        index = 0
        for strategy, size in zip(self.q_learner_groups.keys(), sizes):
            self.q_learner_groups[strategy] = q_learner[index:index+size]
            index += size

        subgroups = {
            #"defectnotbad": strat_defectnotbad + self.q_learner_groups["IV_NNN"],
            #"defectgood": strat_defectgood + self.q_learner_groups["V_NNN"],
            "lonernotbad": strat_lonernotbad + self.q_learner_groups["VI_NNN"],
            "lonergood": strat_lonergood + self.q_learner_groups["VII_NNN"]
        }

        return subgroups

    def _reset_population(self):
        """
        Reset all agents and the population between time-steps.

        Things that are reset:
            An agent's utility is reset to 1

        Things that do not reset:
            Any population parameters
        """

        # logging.info("Agents reset")
        for agent in self.agents:
            agent.utility = 1
            #agent.reputation = 1

    def __str__(self):
        s = []
        for agent in self.agents:
            s += [
                f"A(ID={agent.ID}, s={agent.strategy['ID']}, u={round(agent.utility, 4)}, r={agent.reputation[-1]})"
            ]
        output = "\n".join(s)
        return output
    