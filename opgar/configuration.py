import pandas as pd
from .strategy import _Strategy
from .norm import _Norm

class Configuration:
    """Provides the functionality and error checking for all parameterizations of simulations.

        Args:
            N (int): The number of players in the population.
            n (int): The number of players per Public Goods Game (PGG).
            t (int): The length of the simulation.
            r (float): The growth factor to the group contribution in the PGG, used for rewarding cooperators (r > 1).
            sigma (float): The growth factor for a loner's utility, used to calculate loners' payoff (0 < sigma < (r-1)c, c is the cost sustained by cooperators in every single game)
            composition (dict): A dictionary of strategy and proportion key-value pairs. See _Strategy documentation.
            norm (str): The specific social norm within the population. See _Norm documentation.
            gamma (float): The cost required to punish someone.
            beta (float): The penalty one pays if one is punished.
            m (float): Probability of evolutionary mixing in group selection
            epsilon (float): The rate of mutation under group selection, this is unused with Rand and Nowak evolutionary mechanism.
            strategy_group (str): The name of the model being simulated, see `_Strategy.strategy_groups`.
            omega (float): Number of rounds played of the OPGG in a single time-step. 
            alpha(float): The learning rate for Q-Learning agents.
            discount_factor(float): The discount factor for future rewards in Q-Learning agents.
            exploration_rate(float): The exploration rate for Q-Learning agents, used in epsilon-greedy action selection.
            delta(float): Probability of an agent changing its group.
            minimum_exploration_rate (float): Minimum exploration rate for Q-Learning agents.
            reset_exploration_rate (int): Number of rounds after which the exploration rate is reset to its initial value.
            epsilon_decay(float): Decay rate for exploration rate in Q-Learning agents.

        Returns:
            opgar.Configuration object which is input to opgar.Population object.

        Examples:
            >>> from opgar import Configuration
            >>> config = Configuration.test_case_1()
            >>> config = Configuration(
                    N=1000, 
                    n=5, 
                    t=200000, 
                    r=3, 
                    sigma=1,
                    composition={}.fromkeys(strategies, 1/len(strategies)), 
                    norm="Defector",
                    gamma=1,
                    beta=2,
                    m=0.95,
                    omega=10/11, 
                    epsilon=0.1,
                    strategy_group="all"
                )
    """

    __slots__ = [
        "social_norm",
        "composition",
        "t",
        "r",
        "sigma",
        "n",
        "N",
        #"gamma",
        #"beta",
        #"m",
        #"epsilon",
        "omega",
        "_meta_data",
        "alpha", 
        "discount_factor",
        "exploration_rate",
        "delta",
        "minimum_exploration_rate",
        "reset_exploration_rate",
        "epsilon_decay",
    ]

    def __init__(
        self,
        N: int,
        n: int,
        r: float,
        sigma: float,
        t: int,
        composition: dict,
        norm: str,
        #gamma: float,
        #beta: float,
        #m: float,
        #epsilon: float,
        strategy_group: str, 
        omega: int,
        alpha: float,
        discount_factor: float,
        exploration_rate: float,
        delta: float,
        minimum_exploration_rate: float,
        reset_exploration_rate: int,
        epsilon_decay: float,
    ):
        self._meta_data = {}

        # ----------------------------------------------------------------------
        # SOCIAL NORM (Anti-defector, Anti-loner, Anti-Neither, None)
        # ----------------------------------------------------------------------

        if norm not in _Norm.social_norm_names:
            raise ValueError(f"Your choice of norm ('{norm}') is invalid.")
        self.social_norm = norm 
        self._meta_data["strategy group"] = strategy_group
        
        # An absence of a social norm can only occur when there are no impure strategies in the population.
        if norm == None:
            for strategy in composition.keys():
                if strategy not in _Strategy.strategy_groups["pure"]:
                    raise ValueError(f"Strategy {strategy} cannot exist without a social norm.")

        # ----------------------------------------------------------------------
        # STRATEGY COMPOSITION
        # ----------------------------------------------------------------------
        if type(composition) == dict:
            if round(sum(composition.values()), 6) != 1:
                raise ValueError(
                    f"Your choice of strategy proportions must sum to 1 (not '{sum(composition.values())}')."
                )
            for strategy in composition.keys():
                if strategy not in _Strategy.all_strategies:
                    raise ValueError(
                        f"Your choice of strategy ('{strategy}') is invalid."
                    )
                if composition[strategy] < 0 or composition[strategy] > 1:
                    raise ValueError(
                        f"Your choice of strategy proportion ('{strategy}': {composition[strategy]}) is invalid."
                    )
        elif type(composition) == str:
            if composition not in _Strategy.strategy_groups:
                raise ValueError(
                    f"Your choice of strategy set ('{composition}') is not valid. See _Strategy.strategy_groups"
                )
        self.composition = pd.Series(composition)
        self._meta_data["norm"] = norm


        # ----------------------------------------------------------------------
        # LENGTH OF SIMULATION
        # ----------------------------------------------------------------------
        if t < 1:
            raise ValueError(f"Your choice of t ('{t}') is invalid.")
        self.t = t

        # ----------------------------------------------------------------------
        # GROUP SYNERGY FACTOR
        # ----------------------------------------------------------------------
        self.r = r

        # ----------------------------------------------------------------------
        # LONER PAYOFFS
        # ----------------------------------------------------------------------
        if strategy_group == "noloner" or strategy_group == "prosocialnoloner":
            self.sigma = None
        else:
            if sigma < 0 or sigma is None:
                raise ValueError(f"Your choice of sigma ('{sigma}') is invalid.")
            self.sigma = sigma

        # ----------------------------------------------------------------------
        # POPULATION SIZE
        # ----------------------------------------------------------------------

        if N < 0:
            raise ValueError(f"Your choice of population size ('{N}') is invalid.")
        self.N = N

        # ----------------------------------------------------------------------
        # OPTIONAL PGG SIZE
        # ----------------------------------------------------------------------

        if n < 0 or n > N:
            raise ValueError(f"Your choice of n ('{n}') is invalid.")
        if not N % n == 0:
            raise ValueError(
                f"PGG size ('{n}') must be a divisor of the population size ('{N}')."
            )
        self.n = n

        # ----------------------------------------------------------------------
        # PUNISHMENT COST & PENALTY
        # ----------------------------------------------------------------------
        """
        if "P" in "".join(composition.keys()):
            if gamma > beta:
                raise ValueError(
                    f"The cost to punish ('{gamma}') should not be greater than the penalty "
                    f"incurred by being punished ('{beta}') ."
                )
            if gamma < 0 or beta < 0:
                raise ValueError(
                    f"The cost to punish ('{gamma}') or the  penalty incurred by being punished "
                    f"('{beta}') cannot be negative."
                )
        self.gamma = gamma
        self.beta = beta
        """

        # ----------------------------------------------------------------------
        # PROBABILITY OF FURTHER GAMES IN SAME PERIOD
        # ----------------------------------------------------------------------
        if omega < 1:
            raise ValueError("Probability of further interactions omega ('{omega}') must be within [1,inf).")
        self.omega = omega

        # ----------------------------------------------------------------------
        # GROUP-WIDE vs POPULATION-WIDE EVOLUTION 
        # ----------------------------------------------------------------------
        # If m=0, people always imitate people from outside of the group
        # if m=1, people always imitate people from the same group
        """
        if m < 0 or m > 1:
            raise ValueError("Probability of mutation vs evolution m ('{m}') must be in [0,1].")
        self.m = m
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Probability of mutation epsilon ('{epsilon}') must be in [0,1].")
        self.epsilon = epsilon
        """

        # -----------------------------------------------------------------------
        # Q-LEARNING PARAMETERS
        # -----------------------------------------------------------------------

        if discount_factor < 0 or discount_factor > 1:
            raise ValueError("Discount factor ('{discount_factor}') must be in [0,1].")
        self.discount_factor = discount_factor

        if alpha <= 0 or alpha > 1:
            raise ValueError("Learning rate alpha ('{alpha}') must be in ]0,1].")
        self.alpha = alpha

        if exploration_rate < 0 or exploration_rate > 1:
            raise ValueError("Exploration rate epsilon ('{exploration_rate}') must be in [0,1].")
        self.exploration_rate = exploration_rate

        if reset_exploration_rate < 0 or reset_exploration_rate > t * omega:
            raise ValueError("Reset exploration rate ('{reset_exploration_rate}') must be in [0, t * omega].")
        self.reset_exploration_rate = reset_exploration_rate

        if minimum_exploration_rate < 0 or minimum_exploration_rate > 1:
            raise ValueError("Minimum exploration rate ('{minimum_exploration_rate}') must be in [0,1].")
        self.minimum_exploration_rate = minimum_exploration_rate

        if epsilon_decay < 0 or epsilon_decay > 1: 
            raise ValueError("Epsilon decay ('{epsilon_decay}') must be in [0,1].")
        self.epsilon_decay = epsilon_decay

        # ------------------------------------------------------------------------
        # PROBABILITY OF AN AGENT TO CHANGE ITS BELONGING GROUP
        # ------------------------------------------------------------------------
        if delta < 0 or delta > 1:
            raise ValueError(f"Parameter delta ('{delta}') must be in [0,1].")
        self.delta = delta

    def to_dict(self, rng_seed = None):
        """
        Return the parameters of a configuration object in dictionary.
        """
        config_dict = {slot: getattr(self, slot) for slot in self.__slots__} | {"seed" : rng_seed if rng_seed is not None else {}}
        # Custom strategy group
        try:
            config_dict["composition"] = (
                self._meta_data["strategy group"],
                1 / len(_Strategy.strategy_groups[self._meta_data["strategy group"]]),
            )
        except:
            config_dict["composition"] = (self._meta_data["strategy group"])
        return config_dict

    

