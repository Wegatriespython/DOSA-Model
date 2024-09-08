from mesa import Agent
import numpy as np
from Utilities.Config import Config
from Utilities.utility_function import maximize_utility
from Utilities.expectations import get_market_demand

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employers = {}  # Dictionary to store employers and corresponding working hours
        self.total_working_hours = 0
        self.max_working_hours = 16
        self.consumption_check = 0
        self.dissatistifaction = 0
        self.wage = model.config.MINIMUM_WAGE
        self.income = 0
        self.savings = self.model.config.INITIAL_SAVINGS
        self.optimals = []
        self.got_paid = False
        self.working_hours = 0
        self.leisure = 16
        self.expected_price = model.config.INITIAL_PRICE
        self.expected_wage = model.config.MINIMUM_WAGE
        self.skills = model.config.INITIAL_SKILLS
        self.consumption = 0
        self.desired_consumption = 1
        self.desired_savings = 0
        self.price = model.config.INITIAL_PRICE
        self.price_history = [model.config.INITIAL_PRICE]
        self.MIN_CONSUMPTION = 1
        self.wage_history = [model.config.MINIMUM_WAGE] * 5
        self.mode = 'decentralized'

    def step(self):

        self.update_utilty()
        self.update_expectations()
        self.update_skills()
    def update_utilty(self):


        if self.model.step_count > 0:

          quantity, expected_wage= get_market_demand(self, 'labor')
          demand, prices = get_market_demand(self, 'consumption')
          self.expected_wage = expected_wage
          wage = np.full(self.model.config.TIME_HORIZON, expected_wage)
          prices = np.full(self.model.config.TIME_HORIZON, prices)
          results = maximize_utility(self.savings, wage, prices)
          self.desired_consumption, self.working_hours, self.leisure, self.desired_savings = [arr[0] for arr in results]

          self.optimals = [self.desired_consumption, self.working_hours, self.desired_savings]

          print(f"{self.optimals}")

    def update_expectations(self):

      if self.model.step_count > 0:
        wage_non_zero = list(filter(lambda x: x != 0, self.wage_history)) if any(self.wage_history) else []
        wage_avg = sum(wage_non_zero) / len(wage_non_zero) if wage_non_zero else 0
        demand, prices = get_market_demand(self, 'consumption')
        print(f"consumption {self.consumption}, desired consumption {self.desired_consumption},check {self.consumption_check}")

        if self.consumption < self.desired_consumption:
            self.expected_price *= 1.05
            self.expected_price = min(self.expected_price, self.get_max_consumption_price())
            print("expected_price", self.expected_price)
        else:
            self.expected_price *= 1
        if self.total_working_hours < self.optimals[1]:
            self.expected_wage *= .95
            self.expected_wage = max(self.expected_wage, self.model.config.MINIMUM_WAGE)
        else:
            self.expected_wage *= 1.05
        self.consumption = 0
        self.consumption_check = 0

    def get_min_wage(self):
        check_wage = max(self.model.config.MINIMUM_WAGE, self.wage)
        min_wage = min(check_wage, self.expected_wage)
        return min_wage

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
            self.total_working_hours = self.total_working_hours
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
                self.wage_history.append(self.wage)
                self.income = sum(emp['wage'] * emp['hours'] for emp in self.employers.values())
        else:
            self.wage = 0
            self.income = 0


    def consume(self, quantity, price):
        total_cost = quantity * price
        self.price_history.append(price)
        self.consumption += quantity
        self.consumption_check += quantity
        print(f"consumption {self.consumption}, check {self.consumption_check}")
        self.savings -= total_cost
        self.savings = max(0, self.savings) #prevent negative savings

    def get_max_consumption_price(self):

        amt = self.savings/20 + (self.wage * 16)

        if amt < self.savings:
          return amt
        else:
          return self.savings # Willing to pay up to 10% more than expected

    def get_paid(self, wage):
        self.savings += wage

    def available_hours(self):
        if self.total_working_hours >= self.max_working_hours:
            return 0
        elif self.working_hours + self.total_working_hours >= self.max_working_hours:
            return max(0, self.max_working_hours - self.total_working_hours)
        else:
            return self.working_hours
