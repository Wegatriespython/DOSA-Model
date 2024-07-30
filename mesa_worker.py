from mesa import Agent
import numpy as np

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employed = False
        self.employer = None
        self.wage = model.config.INITIAL_WAGE
        self.savings = model.config.INITIAL_SAVINGS
        self.skills = model.config.INITIAL_SKILLS
        self.consumption = model.config.INITIAL_CONSUMPTION
        self.price_history = [model.config.INITIAL_PRICE]
        self.wage_history = [model.config.MINIMUM_WAGE] * 5

    def step(self):
        self.update_expectations()
        self.make_economic_decision()
        self.update_skills()

    def update_expectations(self):
        self.update_wage_expectation()
        self.update_price_expectation()

    def update_wage_expectation(self):
        if self.employed:
            self.wage_history.append(self.wage)
        else:
            self.wage_history.append(0)  # Represent unemployment with 0 wage
        self.wage_history = self.wage_history[-5:]  # Keep only the last 5 periods
        self.expected_wage = max(np.mean(self.wage_history), self.model.config.MINIMUM_WAGE)

    def update_price_expectation(self):
        if self.seller_prices:
            current_price = np.mean(self.seller_prices)  # Average of current seller prices
            self.price_history.append(current_price)
            if len(self.price_history) > 10:
                self.price_history.pop(0)
            self.expected_price = np.mean(self.price_history)
        else:
            # Fallback to global average if no seller prices are available
            current_price = self.model.global_accounting.get_average_consumption_good_price()
            self.price_history.append(current_price)
            if len(self.price_history) > 10:
                self.price_history.pop(0)
            self.expected_price = np.mean(self.price_history)

    def make_economic_decision(self):
        # Simplified decision-making process
        self.consumption = min(self.savings, self.expected_wage * self.model.config.CONSUMPTION_PROPENSITY)
        self.wage = max(self.expected_wage, self.model.config.MINIMUM_WAGE)

    def update_skills(self):
        if self.employed:
            self.skills *= (1 + self.model.config.SKILL_GROWTH_RATE)
        else:
            self.skills *= (1 - self.model.config.SKILL_DECAY_RATE)

    def get_hired(self, employer, wage):
        self.employed = True
        self.employer = employer
        self.wage = wage

    def get_fired(self):
        self.employed = False
        self.employer = None
        self.wage = 0

    def consume(self, quantity, price):
        total_cost = quantity * price
        self.consumption = quantity
        self.savings -= total_cost

    def get_max_consumption_price(self):
        return self.expected_price * 1.1  # Willing to pay up to 10% more than expected

    def update_after_markets(self):
        if self.employed:
            self.savings += self.wage
        self.savings -= self.consumption * self.model.global_accounting.get_average_consumption_good_price()

    def set_seller_prices(self, prices):
        """
        Set the current prices from sellers in the consumption market.
        
        :param prices: List of prices from sellers in the consumption market
        """
        self.seller_prices = prices