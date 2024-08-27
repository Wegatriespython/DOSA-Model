from mesa import Agent
import numpy as np
from Utilities.Config import Config

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employers = {}  # Dictionary to store employers and corresponding working hours
        self.total_working_hours = 0
        self.max_working_hours = 16
        self.dissatistifaction = 0
        self.wage = 0.0625
        self.savings = 100
        self.got_paid = False
        self.expected_price = model.config.INITIAL_PRICE
        self.expected_wage = model.config.MINIMUM_WAGE
        self.skills = model.config.INITIAL_SKILLS
        self.consumption = 1
        self.desired_consumption = 1
        self.price = model.config.INITIAL_PRICE
        self.price_history = [model.config.INITIAL_PRICE]
        self.MIN_CONSUMPTION = 1
        self.wage_history = [model.config.MINIMUM_WAGE] * 5
        self.mode = 'decentralized'

    def step(self):

        self.update_expectations()
        self.make_economic_decision()
        self.update_skills()



    def update_expectations(self):
        self.update_average_wage()
        self.update_wage_expectation()
        self.update_price_expectation()

    def update_wage_expectation(self):
        if self.total_working_hours >= 12:
            self.expected_wage = self.expected_wage * 1.1 # Hardcoding a wage cieling of 10
            self.expected_wage = min(self.expected_wage, 10)
        else :
            self.expected_wage -= (self.expected_wage - self.model.config.MINIMUM_WAGE) * 0.1 # Hardcoding a wage floor of 10
            self.expected_wage = max(self.expected_wage, self.model.config.MINIMUM_WAGE)

    def update_price_expectation(self):
        self.expected_price = self.model.get_average_consumption_good_price() * 1.1

    def make_economic_decision(self):
        self.desired_consumption = min(self.savings, self.wage * self.total_working_hours * self.model.config.CONSUMPTION_PROPENSITY)
        self.desired_consumption = max(self.desired_consumption, self.model.config.MIN_CONSUMPTION)
        #print(f"Worker {self.unique_id} desired consumption: {self.desired_consumption}")

    def update_skills(self):
        if self.total_working_hours > 0:
            self.skills *= (1 + self.model.config.SKILL_GROWTH_RATE)
        else:
            self.skills *= (1 - self.model.config.SKILL_DECAY_RATE)

    def get_hired(self, employer, wage, hours):
        self.employers[employer] = {'hours': hours, 'wage': wage}
        self.total_working_hours += hours
        self.update_average_wage()

    def update_hours(self, employer, hours):
        if employer in self.employers:
            self.employers[employer]['hours'] += hours
            self.total_working_hours += hours
            self.update_average_wage()

    def get_fired(self, employer):
        if employer in self.employers:
            self.total_working_hours -= self.employers[employer]['hours']
            self.total_working_hours = max(0, self.total_working_hours)
            del self.employers[employer]
            self.update_average_wage()

    def update_average_wage(self):
        if self.total_working_hours > 0:
                self.wage = sum(emp['wage'] * emp['hours'] for emp in self.employers.values()) / self.total_working_hours
        else:
            self.wage = 0

    def consume(self, quantity, price):
        total_cost = quantity * price
        self.price_history.append(price)
        self.consumption = quantity
        self.dissatistifaction = min(0, self.desired_consumption - self.consumption)
        self.savings -= total_cost

    def get_max_consumption_price(self):
        return self.expected_price * 1.1  # Willing to pay up to 10% more than expected

    def get_paid(self, wage):
        self.savings += wage



    def available_hours(self):
        return max(0, self.max_working_hours - self.total_working_hours)
