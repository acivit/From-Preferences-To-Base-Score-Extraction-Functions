import sympy as sp


class Argument:
    def __init__(self, id, base_score=0.5) -> None:
        self.id = id
        self.attackers = []
        self.supporters = []
        self.attacking = []
        self.supporting = []
        self.base_score = base_score
        self.symbol = sp.symbols(id)
        self.energy = sp.symbols(f"E_{id}")
        self.final_strength = None

    def add_node_attacker(self, attacker):
        self.attackers.append(attacker)

    def add_node_supporter(self, supporter):
        self.supporters.append(supporter)

    def add_node_attacking(self, attacking):
        self.attacking.append(attacking)

    def add_node_supporting(self, supporting):
        self.supporting.append(supporting)
