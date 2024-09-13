from mesa import Agent
import numpy as np
from Utilities.Config import Config
from Utilities.utility_function import maximize_utility
from Utilities.expectations import  get_market_demand, expect_price_ar

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employers = {}  # Dictionary to store employers and corresponding working hours
        self.total_working_hours = 0
        self.max_working_hours = 16
        self.worker_expectations = []
        self.consumption_check = 0
        self.wage_history1 = []
        self.price_history1 = []
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
        self.prices = []
        self.MIN_CONSUMPTION = 1
        self.wage_history = [model.config.MINIMUM_WAGE]
        self.mode = 'decentralized'

    def step(self):

        self.update_utilty()
        self.update_expectations()
        self.update_skills()
    def update_utilty(self):


        if self.model.step_count > 1:

          quantity, mkt_wages= get_market_demand(self, 'labor')
          demand, price = get_market_demand(self, 'consumption')
          self.wage_history1.append(mkt_wages)
          self.price_history1.append(price)

          wage = expect_price_ar(self.wage_history1, mkt_wages, self.model.config.TIME_HORIZON)
          prices = expect_price_ar(self.price_history1, price, self.model.config.TIME_HORIZON)

          self.worker_expectations = [wage[0], prices[0]]

          results = maximize_utility(self.savings, wage, prices,0.95, self.model.config.TIME_HORIZON, alpha=0.9, max_working_hours=16)
          self.desired_consumption, self.working_hours, self.leisure, self.desired_savings = [arr[0] for arr in results]

          self.optimals = [self.desired_consumption, self.working_hours, self.desired_savings]



    def update_expectations(self):

      if self.model.step_count > 1:
        if len(self.prices) > 0 :
          self.prices = [p for p in self.prices if not np.isnan(p)]
          avg_price = np.mean(self.prices)
        else:
          avg_price = self.worker_expectations[1]
        if self.consumption < self.desired_consumption:
          # If the worker is consuming less than desired, increase the expected price
            if self.expected_price > avg_price:
              self.expected_price = self.expected_price + (self.get_max_consumption_price() - self.expected_price) * 0.5
            else:
              self.expected_price = min((avg_price + (avg_price - self.expected_price) * 0.5), self.get_max_consumption_price())

        elif self.expected_price < avg_price:
          # if consuming sufficient yet, overpaying, then round 2 clearing is happening, worker needs to increase bid to lower prices.
            self.expected_price = avg_price - (avg_price - self.expected_price) * 0.5
        else:
          # if consuming and in round1 then worker can bargain by lowering the bid
            self.expected_price = avg_price - (self.expected_price - avg_price) * 0.2



        if self.wage < self.expected_wage:
            self.expected_wage *= .95
            self.expected_wage = max(self.expected_wage, self.model.config.MINIMUM_WAGE)
        else:
            self.expected_wage *= 1.05

        self.consumption = 0
        self.consumption_check = 0
        self.price = 0

    def get_min_wage(self):
        min_wage = self.model.config.MINIMUM_WAGE
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
    def quit(self):

        # Sort employers by hours worked, in descending order
        sorted_employers = sorted(self.employers.items(), key=lambda x: x[1]['hours'], reverse=True)

        for employer, details in sorted_employers:
            if self.total_working_hours <= self.working_hours:
                break

            hours_to_quit = min(details['hours'], self.total_working_hours - self.working_hours)

            if hours_to_quit == details['hours']:
                # Quit the job entirely
                self.get_fired(employer)
            else:
                # Reduce hours for this job
                self.employers[employer]['hours'] -= hours_to_quit
                self.total_working_hours -= hours_to_quit

            if self.total_working_hours <= self.working_hours:
                break

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
        self.price = price
        self.prices.append(price)

        self.savings -= total_cost
        self.savings = max(0, self.savings) #prevent negative savings

    def get_max_consumption_price(self):

        amt = self.savings/(self.model.config.TIME_HORIZON) + (self.wage * 16)

        if amt < self.savings:
          return amt
        else:
          return self.savings # Willing to pay up to 10% more than expected

    def get_paid(self, wage):
        self.savings += wage

    def available_hours(self):
        if self.total_working_hours >= self.max_working_hours:
            return 0
        elif self.working_hours > self.total_working_hours:
            return self.working_hours - self.total_working_hours
        else :
          self.quit()
          return 0
