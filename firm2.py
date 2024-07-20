from Basefirm import BaseFirm
from Config import config
import random

class Firm2(BaseFirm):
    def __init__(self):
        super().__init__()
        self.capital = config.FIRM2_INITIAL_CAPITAL
        self.investment = config.FIRM2_INITIAL_INVESTMENT
        self.investment_demand = config.FIRM2_INITIAL_INVESTMENT_DEMAND
        self.desired_capital = config.FIRM2_INITIAL_DESIRED_CAPITAL
        self.machine_output_per_period = config.FIRM2_MACHINE_OUTPUT_PER_PERIOD
        self.inventory_threshold = config.FIRM2_INVENTORY_THRESHOLD

    def produce(self):
        self.production = self.capital * self.productivity * len(self.workers) * config.PRODUCTION_FACTOR
        self.inventory += self.production

    def plan_production(self, workers):
        total_worker_consumption = sum([worker.wage for worker in workers])
        self.demand = total_worker_consumption * (0.9 + 0.2 * random.random())
        self.desired_capital = self.demand * 1.1
        

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
        self.invest()
        if self.demand > self.production:
            self.desired_workers = min(
                (len(self.workers) * 1.1) + 1, int(self.demand / (self.productivity * config.PRODUCTION_FACTOR)))
        elif self.demand < self.production * 0.9:
            self.desired_workers = max(1, int(self.workers * 0.9))
        