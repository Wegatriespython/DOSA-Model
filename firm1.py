from Basefirm import BaseFirm
from Config import config
import random

class Firm1(BaseFirm):
    def __init__(self):
        super().__init__()
        self.capital = config.FIRM1_INITIAL_CAPITAL
        self.RD_investment = config.FIRM1_INITIAL_RD_INVESTMENT
        self.inventory_threshold = config.FIRM1_INVENTORY_THRESHOLD

    def innovate(self):
        if random.random() < config.INNOVATION_PROBABILITY:
            self.RD_investment = self.capital * config.FIRM1_RD_INVESTMENT_RATE
            self.productivity *= (1 + config.PRODUCTIVITY_INCREASE)

    def update_state(self, firm2s):
        super().update_state()
        self.demand = sum([firm.investment_demand for firm in firm2s])

    def fulfill_order(self, quantity):
        sold = min(quantity, self.inventory)
        self.inventory -= sold
        return sold