'''
Example script to run a single simulation of the optional public goods game with a population of agents.
This script sets up a population with a specific configuration and runs the simulation, printing the results.
It uses the `opgar` library to create the population and manage the simulation process.
'''

from opgar import Population, _Strategy, Configuration
import logging


logging.basicConfig(
    level=logging.INFO,                 # Set logging level to INFO 
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True                          
)

only_cooperate_and_learner = _Strategy.strategy_groups["UnconditionalCooperator + Q-Learning"]
only_defect_and_learner = _Strategy.strategy_groups["UnconditionalDefector + Q-Learning"]
only_qlearner = _Strategy.strategy_groups["Q-Learning no punishment"]

config = Configuration(
    N = 1000, 
    n = 10, 
    t = 1000, 
    r = 3, 
    sigma = 1,
    composition = {}.fromkeys(only_qlearner, 1/len(only_qlearner)), #only Q-Learners
    norm = None,
    gamma = 0,
    beta = 0,
    m = 0.95,
    omega = 0, 
    epsilon = 0,
    strategy_group = "Q-Learning no punishment",
    alpha = 0.1,
    exploration_rate = 1.0,
    discount_factor = 0.1
    )

model = Population(config)

results = model.simulate(job_id=3, rng_seed=42)

print(results)
