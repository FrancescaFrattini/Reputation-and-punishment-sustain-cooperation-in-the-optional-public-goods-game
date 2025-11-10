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
only_qlearner = _Strategy.strategy_groups["Q-Learning no punishment"]

config = Configuration(
    N = 500,
    n = 5, 
    t = 500, 
    r = 3, 
    sigma = 1,
    composition = dict(zip(only_loner_and_learner, [0.75, 0.25])),
    norm = None,
    omega = 10, 
    strategy_group = "UnconditionalLoner + Q-Learning",
    alpha = 0.1,
    exploration_rate = 1.0,
    discount_factor = 0.9,
    delta = 1.0,
    minimum_exploration_rate = 0.05,
    reset_exploration_rate = None,
    epsilon_decay = 0.999
    )

for id in range(0, 1):
    seed = random.randint(0, 1000)
    random.seed(seed)
    model = Population(config)
    np.random.seed(seed)
    results = model.simulate(job_id=id, rng_seed=seed, transition_matrix_batch=50)
