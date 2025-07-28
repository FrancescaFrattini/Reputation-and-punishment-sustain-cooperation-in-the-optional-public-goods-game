class _Norm:
    """
    A population-wide object specifying the rules of reputation assignment after a round of the Public Goods Game. 
    The population updates agents' reputations following any round of the OPGG with _Norm.assign_reputation taking 
    an action (one of 1, 0, None representing C, D, L) and returning 1, 0, -1 representing a good, intermediate and bad reputation. 
    This reputation map is selected from `_Norm.social_norm_rules` and stored in `_Norm.reputation_matrix`.

    Class variables
        - Use `_Norm.social_norm_names` fpr a list of the social norm names
        - Use `_Norm.social_norm_rules` for the dictionary of actions: reputation key/value pairs
        - Use `_Norm.social_norm_descriptions` for an explanation of each norm

    Args:
        norm_name (str) = Name of the social norm of the population, can be one of ["Loner", "Defector", "Neither", \
        "Both", None].
    """
    
    '''
    1 -> Cooperate
    0 -> Defect
    None -> Lone
    '''
    social_norm_rules = {
        "Loner": {1: 1, 0: 0, None: -1},
        "Defector": {1: 1, 0: -1, None: 0},
        "Neither": {1: 1, 0: 0, None: 0},
        "Both": {1: 1, 0: -1, None: -1},
        None: None,
    }

    social_norm_descriptions = {
        "Loner": {
            "Name": "Loner",
            "Summary": "This is the social norm that is biased against people who do not contribute.",
            "Rule": "Assign 1 (good) if player contributed, 0 (okay) if he defected, -1 (bad) if he did not "
            "participate.",
        },
        "Defector": {
            "Name": "Defector",
            "Summary": "This is the social norm that is biased against people who participate but do not contribute.",
            "Rule": "Assign 1 (good) if player contributed, 0 (okay) if he didn't participate, -1 (bad) if he "
            "participated but did not contribute.",
        },
        "Neither": {
            "Name": "Neither",
            "Summary": "This is the social norm that is indifferent towards Loners or Defectors.",
            "Rule": "Assign 1 (good) if player contributed, 0 (okay) if he did not contributed or if he did not "
            "participate.",
        },
        "Both": {
            "Name": "Both",
            "Summary": "This social norm hates both Loners and Defectors.",
            "Rule": "Assign 1 (good) if player contributed, -1 (bad) if he defected or abstained.",
        },
        None: {
            "Name": "None",
            "Summary": "This is the social norm that ignores reputation.",
            "Rule": "No actions are assigned reputations. This is only valid in populations with only strategies "
            "I, II, III, and XII. Errors will be raised to prevent this.",
        },
    }
    social_norm_names = [*social_norm_descriptions.keys()]

    def __init__(self, norm_name):
        self.norm_name = norm_name
        self.reputation_matrix = _Norm.social_norm_rules[norm_name]

    def _assign_reputation(self, contribution):
        """
        Given a contribution, assign new reputations according to the social norm type
        Args:
            contribution (int or None): An agent's contribution (1) or defection (0) or loner (None)

        Returns:
            A new reputation, either good (1), okay (0), bad (-1)
        """
        return self.reputation_matrix[contribution]
