from mesa import Agent
import numpy as np
from Utilities.Config import Config

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employers = {}  # Dictionary to store employers and corresponding working hours
        self.total_working_hours = 0
        self.max_working_hours = 16
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
        if self.mode == 'decentralized':
            self.decentralized_step()
        elif self.mode == 'centralized':
            self.centralized_step()

    def decentralized_step(self):
        self.update_expectations()
        self.make_economic_decision()
        self.update_skills()

    def centralized_step(self):
        # The central planner will call apply_central_decision()
        self.update_skills()

    def update_expectations(self):
        self.update_wage_expectation()
        self.update_price_expectation()

    def update_wage_expectation(self):
        if self.total_working_hours > 0:
            self.wage_history.append(self.wage)
        else:
            self.wage_history.append(0)
        self.wage_history = self.wage_history[-5:]
        self.expected_wage = max(np.mean(self.wage_history), self.model.config.MINIMUM_WAGE)

    def update_price_expectation(self):
        if len(self.price_history) > 10:
            self.price_history.pop(0)
            self.expected_price = np.mean(self.price_history) * 1.1
        else:
            current_price = self.model.get_average_consumption_good_price()
            self.expected_price = current_price * 1.1

    def make_economic_decision(self):
        self.desired_consumption = min(self.savings, self.expected_wage * self.total_working_hours * self.model.config.CONSUMPTION_PROPENSITY)
        self.desired_consumption = max(self.desired_consumption, self.model.config.MIN_CONSUMPTION)
        #print(f"Worker {self.unique_id} desired consumption: {self.desired_consumption}")

    def update_skills(self):
        if self.total_working_hours > 0:
            self.skills *= (1 + self.model.config.SKILL_GROWTH_RATE)
        else:
            self.skills *= (1 - self.model.config.SKILL_DECAY_RATE)

    def get_hired(self, employer, wage, hours):
        if employer in self.employers:
            self.employers[employer]['hours'] += hours
        else:
            self.employers[employer] = {'hours': hours, 'wage': wage}
        self.total_working_hours += hours
        self.update_average_wage()

    def update_hours(self, employer, hours):
        if employer in self.employers:
            old_hours = self.employers[employer]['hours']
            self.employers[employer]['hours'] += hours
            self.total_working_hours += (hours - old_hours)
            self.update_average_wage()

    def get_fired(self, employer):
        if employer in self.employers:
            self.total_working_hours -= self.employers[employer]['hours']
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
        self.savings -= total_cost

    def get_max_consumption_price(self):
        return self.expected_price * 1.1  # Willing to pay up to 10% more than expected

    def update_after_markets(self):
        self.savings += sum(emp['wage'] * emp['hours'] for emp in self.employers.values())


    def available_hours(self):
        return max(0, self.max_working_hours - self.total_working_hours)


    def apply_central_decision(self, employment, wage, consumption):
        self.employed = employment
        self.wage = wage
        self.consumption = consumption
        if self.employed:
            self.savings += self.wage*self.working_hours
        self.savings -= self.consumption  # Consumption good price is 1 (numeraire)
        if not self.employed:
            self.employer = None
