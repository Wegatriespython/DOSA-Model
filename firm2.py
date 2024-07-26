# firm2.py
from Basefirm import BaseFirm
from Config import config

class Firm2(BaseFirm):
    def __init__(self):
        super().__init__()
        self.capital = config.FIRM2_INITIAL_CAPITAL
        self.investment = config.FIRM2_INITIAL_INVESTMENT
        self.investment_demand = config.FIRM2_INITIAL_INVESTMENT_DEMAND
        self.desired_capital = config.FIRM2_INITIAL_DESIRED_CAPITAL
        self.machine_output_per_period = config.FIRM2_MACHINE_OUTPUT_PER_PERIOD
        self.inventory_threshold = config.FIRM2_INVENTORY_THRESHOLD

    def calculate_desired_capital(self):
        self.desired_capital = self.demand * 1.1  # Desire 10% more capital than current demand

    def invest(self):
        if self.capital < self.desired_capital:
            self.investment_demand = self.desired_capital - self.capital
        else:
            self.investment_demand = 0

    def receive_capital(self, capital_amount):
        self.capital += capital_amount
        self.investment_demand -= capital_amount

    def update_state(self):
        super().update_state()
        self.calculate_desired_capital()
        self.invest()