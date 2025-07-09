from itertools import product


class _Strategy:
    """
    The _Strategy class provides all the relevant behavioural strategy functions for agents deciding their actions. 
    Agents call the _Strategy._choose_action and _Strategy._choose_punishment methods to assign actions and punishments according
    to the function look-up tables this class provides.

    Class variables:
        - `_Strategy.behavioural_strategy_names` is a list of the names of the strategies ignoring punishment variants (ie. the strategy roots).
        - `_Strategy.behavioural_strategy` is a dictionary where the keys are the strategy roots, and the values are additional dictionaries containing the lambda function prescribing the action, and a description of the strategy
        - `_Strategy.punishment_strategies_names` and `_Strategy.punishment_strategies` are the same as above except describing the punishment variants and ignoring the strategy roots. 
        - _Strategy.all_strategies is a comprehensive list of all strategies available under this model
        - _Strategy.strategy_groups is a dictionary containing lists of strategies grouped by particular models. 

            - all - Model with Reputation and Punishment
            - pure - Baseline model with only punishment
            - purenopunish - Baseline model with no reputation or punishment
            - nopunish - Model with only reputation
            - impure - All conditional strategies
            - prosocial - AllC, AllD and AllL that punishes defectors.
            - prosocialnoloner - AllC and AllD that punishes defectors.
            - noloner - All strategies that do not involve being a loner.
            - noAllC - All strategies without any unconditional cooperators.

        - _Strategy.strategy_name_mapping is a dictionary mapping the roman numeral version of strategy names (from I - XI) to the form used in the paper.    
    """

    behavioural_strategy_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    behavioural_strategy = {
        "I": {
            "function": lambda avg: 1,
            "description": "Unconditional Cooperation",
        },
        "II": {
            "function": lambda avg: 0,
            "description": "Unconditional Defection",
        },
        "III": {
            "function": lambda avg: None,
            "description": "Never participate",
        },
        "IV": {
            "function": lambda avg: 1 if avg > -1 else 0,
            "description": "Cooperate if anyone in the group is not bad otherwise defect",
        },
        "V": {
            "function": lambda avg: 1 if avg > 0 else 0,
            "description": "Cooperate if group has mostly good people otherwise defect",
        },
        "VI": {
            "function": lambda avg: 1 if avg > -1 else None,
            "description": "Cooperate if anyone in the group is not bad otherwise don't participate",
        },
        "VII": {
            "function": lambda avg: 1 if avg > 0 else None,
            "description": "Cooperate if group has mostly good people otherwise don't participate",
        },
        "VIII": {
            "function": lambda avg: 0 if avg > -1 else None,
            "description": "Defect if group has any not-bad people otherwise don't participate",
        },
        "IX": {
            "function": lambda avg: 0 if avg > 0 else None,
            "description": "Defect if group is mostly good otherwise don't participate",
        },
        "X": {
            "function": lambda avg: None if avg > -1 else 0,
            "description": "Don't participate if group has anyone not-bad otherwise defect",
        },
        "XI": {
            "function": lambda avg: None if avg > 0 else 0,
            "description": "Don't participate if group is mostly good otherwise defect",
        },
    }

    punishment_strategies_names = ["".join(d) for d in product("NP", repeat=3)]
    punishment_strategies = {
        "NNN": {"function": lambda action: False, "description": "Do not punish"},
        "NNP": {
            "function": lambda action: True if action in [None] else False,
            "description": "Punish loners",
        },
        "NPN": {
            "function": lambda action: True if action in [0] else False,
            "description": "Punish defectors",
        },
        "NPP": {
            "function": lambda action: True if action in [0, None] else False,
            "description": "Punish defectors and loners",
        },
        "PNN": {
            "function": lambda action: True if action in [1] else False,
            "description": "Punish cooperators",
        },
        "PNP": {
            "function": lambda action: True if action in [1, None] else False,
            "description": "Punish cooperators and loners",
        },
        "PPN": {
            "function": lambda action: True if action in [1, 0] else False,
            "description": "Punish cooperators and defectors",
        },
        "PPP": {"function": lambda action: True, "description": "Punish everyone"},
    }

    all_strategies = [
        "_".join([i, j])
        for i, j in product(behavioural_strategy_names, punishment_strategies_names)
    ]
    strategy_groups = {
        "all": all_strategies,
        "pure": all_strategies[:24],
        "purenopunish": all_strategies[:24:8],
        "nopunish": [strategy for strategy in all_strategies if "_NNN" in strategy],
        "impure": all_strategies[24:],
        "prosocial": ["I_NNN", "I_NPN", "II_NNN", "III_NNN"],
        "prosocialnoloner": ["I_NNN", "I_NPN", "II_NNN"],
        "noloner": [strategy for strategy in all_strategies if strategy.split("_")[0] in ["I", "II", "IV", "V"]],
        "noAllC": all_strategies[8:],
    }

    strategy_name_mapping = {
        "I": "I",
        "II": "II",
        "III": "III",
        "IV": "I^{-1,D}",
        "V": "I^{0,D}",
        "VI": "I^{-1,L}",
        "VII": "I^{0,L}",
        "VIII": "II^{-1,L}",
        "IX": "II^{0,L}",
        "X": "III^{-1,D}",
        "XI": "III^{0,D}",
    }

    @staticmethod
    def _choose_action(ID, avg_rep):
        """
        Return a function implementing the agent's behavioural strategy depending on the strategy from _Strategy.names.
        Each agent stores a reference to their function in _Strategy.choices to call as needed.

        Args:
            ID (str): A string representation the 11 behavioural strategies 'I' - 'XI' within the model
            avg_rep (float): The average reputation of the PGG group, used as an argument to the strategy behavioural
            function.
        Returns:
            _Strategy.choices[ID] (func): A function requiring a single argument 'average reputation'. See
            agent._choose_action for more information.
        """
        return _Strategy.behavioural_strategy[ID]["function"](avg_rep)

    @staticmethod
    def _choose_punishment(ID, action):
        """
        Return a function implementing the agent's punishment strategy depending on a receiving agent's last action out
        of 1, 0, or None representing a contribution, defection or withdrawal from the public goods game.

        Args:
            ID (str): A string representation of the punishment strategy 'NNN', 'NNP', ..., 'PPP'. These denote
            punishment in the order of Cooperator, Defector, Loner such that 'PPN' entails punishment towards
            cooperators and defectors but not loners.
            action ([1, 0, None]): An action specifying whether the receiving agent's last action was cooperation,
            defection or withdrawal.
        Returns:
            punishment (bool): A True/False value to state whether the receiving agent will or will not receive
            punishment from this agent.
        """
        return _Strategy.punishment_strategies[ID]["function"](action)
