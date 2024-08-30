from mesa import Agent
import numpy as np
from Utilities.Config import Config
from Utilities.utility_function import maximize_utility

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
        self.desired_savings = 0
        self.price = model.config.INITIAL_PRICE
        self.price_history = [model.config.INITIAL_PRICE]
        self.MIN_CONSUMPTION = 1
        self.wage_history = [model.config.MINIMUM_WAGE] * 5
        self.mode = 'decentralized'

    def step(self):

        self.update_utilty()
        self.update_skills()
    def update_utilty(self):
        if self.model.step_count > 10:
            print(f"Calling maximize utility with savings: {self.savings}, wage: {self.wage}, average consumption good price: {self.model.get_average_consumption_good_price()}, price history: {self.price_history[-1]}, total working hours: {self.total_working_hours}")
        wage = max(self.wage, self.model.config.MINIMUM_WAGE)
        results = maximize_utility(self.savings, wage, self.model.get_average_consumption_good_price(), self.price_history[-1], self.total_working_hours)
        self.desired_consumption, self.total_working_hours, self.desired_savings = results
        self.total_working_hours = round(self.total_working_hours)
        self.desired_consumption = round(self.desired_consumption)
        self.desired_savings = round(self.desired_savings)
        if self.savings < self.desired_consumption:
            self.desired_consumption = 0
        if self.savings < self.desired_savings:
            self.expected_wage = self.model.get_average_wage() * 1.1
        if self.consumption < self.desired_consumption:
            self.expected_price = self.model.get_average_consumption_good_price() * 1.1
    def get_min_wage(self):
        return max(self.model.config.MINIMUM_WAGE, self.wage)

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
            self.total_working_hours = round(self.total_working_hours)
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
        self.savings = max(0, self.savings) #prevent negative savings

    def get_max_consumption_price(self):
        return self.expected_price * 1.1  # Willing to pay up to 10% more than expected

    def get_paid(self, wage):
        self.savings += wage



    def available_hours(self):
        return max(0, round(self.max_working_hours) - round(self.total_working_hours))
