import pandas as pd
from .strategy import _Strategy
from .norm import _Norm

class Configuration:
    """Provides the functionality and error checking for all parameterizations of simulations.

        Args:
            N (int): The number of players in the population.
            n (int): The number of players per Public Goods Game (PGG).
            t (int): The length of the simulation.
            r (float): The growth factor to the group contribution in the PGG.
            sigma (float): The growth factor for a loner's utility.
            composition (dict): A dictionary of strategy and proportion key-value pairs. See _Strategy documentation.
            norm (str): The specific social norm within the population. See _Norm documentation.
            gamma (float): The cost required to punish someone.
            beta (float): The penalty one pays if one is punished.
            u (float): The probability of update to a random strategy in the strategy group using the Rand and Nowak evolutionary mechanism.
            m (float): The degree of evolutionary mixing in group selection
            epsilon (float): The rate of mutation under group selection, this is unused with Rand and Nowak evolutionary mechanism.
            strategy_group (str): The name of the model being simulated, see `_Strategy.strategy_groups`.
            omega (float): Probability of multiple rounds of the OPGG in a single time-step. 

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
                    u=0.01,
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
        "gamma",
        "beta",
        "u",
        "m",
        "epsilon",
        "omega",
        "_meta_data",
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
        gamma: float,
        beta: float,
        u: float,
        m: float,
        epsilon: float,
        strategy_group: str, 
        omega: float = 0,
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

        # ----------------------------------------------------------------------
        # EVOLUTIONARY UPDATE PROBABILITY
        # ----------------------------------------------------------------------
        self.u = u

        # ----------------------------------------------------------------------
        # PROBABILITY OF FURTHER GAMES IN SAME PERIOD
        # ----------------------------------------------------------------------
        if omega < 0 or omega >= 1:
            raise ValueError("Probability of further interactions omega ('{omega}') must be within [0,1).")
        self.omega = omega

        # ----------------------------------------------------------------------
        # GROUP-WIDE vs POPULATION-WIDE EVOLUTION 
        # ----------------------------------------------------------------------
        # If m=0, people always imitate people from outside of the group
        # if m=1, people always imitate people from the same group
        if m < 0 or m > 1:
            raise ValueError("Probability of mutation vs evolution m ('{m}') must be in [0,1].")
        self.m = m
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Probability of mutation epsilon ('{epsilon}') must be in [0,1].")
        self.epsilon = epsilon


    def to_dict(self):
        """
        Return the parameters of a configuration object in dictionary.
        """
        config_dict = {slot: getattr(self, slot) for slot in self.__slots__}
        # Custom strategy group
        try:
            config_dict["composition"] = (
                self._meta_data["strategy group"],
                1 / len(_Strategy.strategy_groups[self._meta_data["strategy group"]]),
            )
        except:
            config_dict["composition"] = (self._meta_data["strategy group"])
        return config_dict
