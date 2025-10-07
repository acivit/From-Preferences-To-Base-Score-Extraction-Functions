import numpy as np
from typing import List, Tuple
import re
from src.argument import Argument

from qbaf import QBAFramework, QBAFARelations
from qbaf_visualizer.Visualizer import visualize
import matplotlib.pyplot as plt


class ArgumentationFramework:

    # TODO: Create method to add argument and its relations in the .dl file
    # TODO: Create

    def __init__(self, scoring="linear", small_gap=1, large_gap=3) -> None:
        self.arguments = {}
        self.attacks = []
        self.supports = []
        self.decisions = {}
        self.value_base_scoring = scoring
        self.small_gap = small_gap
        self.large_gap = large_gap
        self.order = None
        self.values_dict = {}

        self.active_arguments = {}
        self.active_attacks = []
        self.active_supports = []
        self.qbaf = None

    def compute_base_score_from_rank(self, rank, total_ranks, a=0, b=0, tmax=1, tmin=0):
        """
        Compute the value based on the rank.
        The value is computed as 1/(rank+1) to ensure that higher ranks have lower values.
        """
        method = self.value_base_scoring
        if method == "from_rank":
            return (total_ranks - rank + 1) / (total_ranks + 1)
        elif method == "from_rank_edged":
            return tmin + (tmax - tmin) * (total_ranks - rank) / (total_ranks - 1)
        elif method == "from_rank_edged_a_b":
            return (total_ranks - rank + a) / (total_ranks - 1 + b)
        elif method == "linear":
            return 1 + (1 - rank) / total_ranks
        elif method == "inverse":
            return 1 / rank
        elif method == "inverse_quadratic":
            return 1 / (rank) ** 2
        elif method == "inverse_cubic":
            return 1 / (rank) ** 3
        elif method == "tanh":
            return (-np.tanh(rank - 2) + 1) / 2
        elif method == "sigmoid":
            return 1 / (1 + np.exp(rank - 2))
        else:
            raise ValueError(
                "Invalid scoring method. Choose 'from_rank', 'linear', 'inverse', 'inverse_quadratic', 'inverse_cubic', 'tanh' or 'sigmoid'."
            )

    def parse_expression(self, line: str) -> List[str]:
        """
        Parse the expression from the line and return a list of tokens.
        The expression is expected to be in the format: "a > b = c >> d"
        where '>' indicates a stronger relation, '=' indicates equality, and '>>' indicates a much stronger relation.
        """
        # Remove whitespace
        expr = line.replace(" ", "")

        # Check if the line is empty
        if not expr:
            return []
        # Check if the line contains only whitespace
        if not expr.strip():
            return []

        # Remove comments
        expr = re.sub(r"#.*", "", expr)

        # Remove any leading or trailing whitespace
        line = expr.strip()

        # Check if the line is a valid expression
        if not re.match(
            r"^[a-zA-Z0-9_]+(\s*>>\s*[a-zA-Z0-9_]+|\s*>\s*[a-zA-Z0-9_]+|\s*=\s*[a-zA-Z0-9_]+)*$",
            line,
        ):
            raise ValueError(f"Invalid expression: {line}")
        # Split into tokens (letters and operators)
        self.order = line
        tokens = re.split(r"(>>|>|=)", line)
        return tokens

    

    def get_values_base_scores_from_order(self, order, a=0, b=0, tmax=1, tmin=0):
        """
        Read the values from the file and compute the base scores for each value.
        The file is expected to contain a single line with the format:
        "a > b = c >> d"
        where '>' indicates a stronger relation, '=' indicates equality, and '>>' indicates a much stronger relation.
        """

        values = {}
        try:
            tokens = self.parse_expression(order)
            if not tokens:
                print("")
                return {}

            # ranking = {}
            # current_level = 1
            # # Split by '>' first to get groups
            # groups = [group.strip() for group in line.split(">")]
            # for group in groups:
            #     items = [item.strip() for item in group.split("=")]
            #     ranking[current_level] = items
            #     current_level += 1

            current_rank_value = 1
            prev_value = tokens[0]
            values[prev_value] = current_rank_value

            i = 1
            while i < len(tokens):
                operator = tokens[i]
                current_value = tokens[i + 1]

                if operator == "=":
                    values[current_value] = values[prev_value]
                elif operator == ">":
                    current_rank_value = values[prev_value] + self.small_gap
                    values[current_value] = current_rank_value
                elif operator == ">>":
                    current_rank_value = values[prev_value] + self.large_gap
                    values[current_value] = current_rank_value

                prev_value = current_value
                i += 2

            total_ranks = current_rank_value

            for value, rank in values.items():
                if total_ranks == 1:
                    base_score = 0.5
                else:
                    base_score = self.compute_base_score_from_rank(
                        rank, total_ranks, a=a, b=b, tmax=tmax, tmin=tmin
                    )
                values[value] = base_score
            print("Order: ", order)
            print("Value base scores", values)
            return values

        except FileNotFoundError:
            print(f"File not found. Setting all values to 0.5.")
            return {}

    def modify_arguments_weights_without_modifying_file(self, args_list):
        for arg, base_score in args_list:
            if arg in self.arguments:
                self.arguments[arg].base_score = base_score
            else:
                print("Argument. ", arg, " -> not in arguments")
                
    def generate_af_from_file(self, framework_text, order=""):
        # Define structures to hold parsed data

        default_base_score = 0.5

        self.values_dict = self.get_values_base_scores_from_order(order=order)
        print("Values dict: ", self.values_dict)
        # print("Values dict: ", self.values_dict)

        # Read and parse the .dl file
        
        for line in framework_text:
            # print("Line read from text area: ", line)
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue

            # Match and parse arguments with or without weight
            arg_match = re.match(r"arg\((\w+)(?:,\s*([\w.]+))?\)", line)
            if arg_match:
                arg_name = arg_match.group(1)
                if arg_match.group(2) is not None:
                    base_score = float(arg_match.group(2))
                else:
                    base_score = self.values_dict.get(arg_name, 0.5)
                self.arguments[arg_name] = Argument(
                    id=arg_name, base_score=base_score
                )
                continue

            # Match and parse arguments with or without weight
            dec_match = re.match(r"dec\((\w+)(?:,\s*([0-9.]+))?\)", line)
            if dec_match:
                arg_name = dec_match.group(1)
                base_score = (
                    float(dec_match.group(2))
                    if dec_match.group(2)
                    else default_base_score
                )
                self.arguments[arg_name] = Argument(
                    id=arg_name, base_score=base_score
                )
                self.decisions[arg_name] = self.arguments[arg_name]
                continue

            # Match and parse attack relations
            att_match = re.match(r"att\((\w+),\s*(\w+)\)", line)
            if att_match:
                attacker = att_match.group(1)
                target = att_match.group(2)
                self.attacks.append((attacker, target))

                self.arguments[attacker].add_node_attacking(target)
                self.arguments[target].add_node_attacker(attacker)

                continue

            # Match and parse support relations
            supp_match = re.match(r"sup\((\w+),\s*(\w+)\)", line)
            if supp_match:
                supporter = supp_match.group(1)
                target = supp_match.group(2)
                self.supports.append((supporter, target))

                self.arguments[supporter].add_node_supporting(target)
                self.arguments[target].add_node_supporter(supporter)
                continue

            self.active_arguments = self.decisions

        self.active_attacks = self.attacks
        self.active_supports = self.supports
        self.active_arguments = self.arguments

    def activate_all_arguments(self):
        self.active_attacks = self.attacks
        self.active_supports = self.supports
        self.active_arguments = self.arguments

    def activate_additional_arguments(self, args):

        for arg in args:
            if arg not in self.active_arguments.keys():
                self.active_arguments[arg] = self.arguments[arg]
                self.active_attacks += [
                    att
                    for att in self.attacks
                    if att[0] == arg and att[1] in self.active_arguments.keys()
                ]
                self.active_attacks += [
                    att
                    for att in self.attacks
                    if att[1] == arg and att[0] in self.active_arguments.keys()
                ]
                self.active_supports += [
                    sup
                    for sup in self.supports
                    if sup[0] == arg and sup[1] in self.active_arguments.keys()
                ]
                self.active_supports += [
                    sup
                    for sup in self.supports
                    if sup[1] == arg and sup[0] in self.active_arguments.keys()
                ]
                self.active_attacks = list(set(self.active_attacks))
                self.active_supports = list(set(self.active_supports))

    def activate_arguments(self, args):

        self.active_arguments = {
            arg: arg_obj for arg, arg_obj in self.arguments.items() if arg in args
        }

        self.active_attacks = [
            att for att in self.attacks if att[0] in args and att[1] in args
        ]
        self.active_supports = [
            sup for sup in self.supports if sup[0] in args and sup[1] in args
        ]

        self.activate_additional_arguments(self.decisions.keys())

        # print("New active_attacks: ", self.active_attacks)
        # print("New active supports: ", self.active_supports)

    def modify_arguments_base_scores_without_modifying_file(self, args_list):
        for arg, base_score in args_list:
            if arg in self.arguments:
                self.arguments[arg].base_score = base_score
            else:
                print("Argument. ", arg, " -> not in arguments")

    def set_all_argument_base_scores_to_value(self, value):

        for arg in self.arguments:
            self.arguments[arg].base_score = value

    def get_final_strengths(self, semantics="QuadraticEnergy_model"):
        """
        Solves qbaf to get final strength of arguments
        semantics: # Nico potyka suggests this semantics in 2019, better than exponent-based semantics https://hal.science/hal-04530784/document
        """
        args = []
        base_scores = []
        for id, arg_object in self.active_arguments.items():
            args.append(id)
            base_scores.append(arg_object.base_score)

        self.qbaf = QBAFramework(
            args,
            base_scores,
            self.active_attacks,
            self.active_supports,
            semantics=semantics,
        )

        return self.qbaf
