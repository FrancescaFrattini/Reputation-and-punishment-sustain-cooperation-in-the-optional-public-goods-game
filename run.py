'''
Example script to run a single simulation of the optional public goods game with a population of agents.
This script sets up a population with a specific configuration and runs the simulation, printing the results.
It uses the `opgar` library to create the population and manage the simulation process.
'''

import random
import numpy as np
from opgar import Population, _Strategy, Configuration
import logging


logging.basicConfig(
    level=logging.INFO,                 # Set logging level to INFO 
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True                          
)

only_cooperate_and_learner = _Strategy.strategy_groups["UnconditionalCooperator + Q-Learning"]
only_defect_and_learner = _Strategy.strategy_groups["UnconditionalDefector + Q-Learning"]
only_loner_and_learner = _Strategy.strategy_groups["UnconditionalLoner + Q-Learning"]
only_qlearner = _Strategy.strategy_groups["Q-Learning"]
defined_strategies_and_learner = _Strategy.strategy_groups["purenopunish + Q-Learning"]
allconditionalcooperator_and_learner = _Strategy.strategy_groups["allconditionalcooperators + Q-Learning"]
conditionalcooperator_and_learner = _Strategy.strategy_groups["cooperatorsdefect + Q-Learning"]
conditionalcooperator1_and_learner = _Strategy.strategy_groups["cooperatorsabstain + Q-Learning"]
goodcooperator_and_learner = _Strategy.strategy_groups["goodcooperators + Q-Learning"]
notbadcooperator_and_learner = _Strategy.strategy_groups["notbadcooperators + Q-Learning"]
conditionaldefector_and_learner = _Strategy.strategy_groups["conditionaldefectors + Q-Learning"]
conditionalloner_and_learner = _Strategy.strategy_groups["conditionalloners + Q-Learning"]

config = Configuration(
    N = 600,
    n = 5, 
    t = 100, 
    r = 3, 
    sigma = 1,
    composition = {}.fromkeys(conditionalcooperator1_and_learner, 1 / len(conditionalcooperator1_and_learner)),
    norm = "Defector",
    omega = 100, 
    strategy_group = "cooperatorsabstain + Q-Learning",
    alpha = 0.1,
    exploration_rate = 1.0,
    discount_factor = 0.9,
    delta = 1.0,
    minimum_exploration_rate = 0.05,
    reset_exploration_rate = None,
    epsilon_decay = 0.9995,
    observation_space = 1
    )

for id in range(0, 10):
    seed = random.randint(0, 1000)
    random.seed(seed)
    model = Population(config)
    np.random.seed(seed)
    results = model.simulate(job_id=id, rng_seed=seed, transition_matrix_batch=10)
