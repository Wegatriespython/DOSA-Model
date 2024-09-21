from sys import breakpointhook
from mesa import Agent
import numpy as np
from Utilities.Config import Config
from Utilities.utility_function import maximize_utility
from Utilities.expectations import  expect_price_trend, get_market_demand, expect_price_ar
from Utilities.Strategic_adjustments import update_worker_price_expectation, update_worker_wage_expectation

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employers = {}  # Dictionary to store employers and corresponding working hours
        self.total_working_hours = 0
        self.preference_mode = self.model.config.PREFERNCE_MODE_LABOR
        self.max_working_hours = 16
        self.worker_expectations = []
        self.skillscarbon = 1
        self.consumption_check = 0
        self.desired_price = self.model.config.INITIAL_PRICE
        self.desired_wage = self.model.config.MINIMUM_WAGE
        self.dissatistifaction = 0
        self.wage_history1 = []
        self.price_history1 = []
        self.avg_price = 0
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
        self.worker_expectations.clear()
        self.update_expectations()
        self.update_utilty()
        self.update_strategy()
        self.update_skills()

    def update_expectations(self):
      if self.model.step_count < 1:
        wage = np.full(self.model.time_horizon,self.expected_wage)
        prices = np.full(self.model.time_horizon, self.expected_price)
        self.worker_expectations = [wage, prices]

        self.avg_price = self.expected_price
      else:
        quantity, mkt_wages= get_market_demand(self, 'labor')
        demand, mkt_price = get_market_demand(self, 'consumption')
        if np.mean(mkt_wages) > self.model.config.MINIMUM_WAGE:
          self.wage_history1.append(mkt_wages)
          self.price_history1.append(mkt_price)
          if len(self.wage_history1)>10:
            self.wage_history1.pop(0)
          if len(self.price_history1)>10:
            self.price_history1.pop(0)

        if len(self.prices) > 0 :
          self.prices = [p for p in self.prices if not np.isnan(p)]
          self.avg_price = np.mean(self.prices)
          self.price_history1.append(self.avg_price)
          if len(self.price_history1)>10:
            self.price_history1.pop(0)
        else:
          self.avg_price = 0


        if self.wage > self.model.config.MINIMUM_WAGE and not np.isnan(self.wage):
          wage = self.wage
        else :
          wage = max(np.mean(mkt_wages), self.model.config.MINIMUM_WAGE)
        if self.avg_price != 0 and not np.isnan(self.avg_price):
          price = self.avg_price
        else:
          price = mkt_price


        wage = expect_price_trend(self.wage_history1, wage, self.model.time_horizon)
        prices = expect_price_trend(self.price_history1, price, self.model.time_horizon)

        self.worker_expectations = [wage, prices]
        print(self.worker_expectations)

    def update_utilty(self):

          Utility_params ={
            'savings' : round(self.savings,2),
            'wage': self.worker_expectations[0],
            'price': self.worker_expectations[1],
            'discount_rate': self.model.config.DISCOUNT_RATE,
            'time_horizon': self.model.time_horizon,
            'alpha': 0.9,
            'max_working_hours': 16
          }
          #print(Utility_params)
          results = maximize_utility(Utility_params)
          self.desired_consumption, self.working_hours, self.leisure, self.desired_savings = [arr[0] for arr in results]

          self.optimals = [self.desired_consumption, self.working_hours, self.desired_savings]
          print(self.optimals)




    def update_strategy(self):

        max_price = self.get_max_consumption_price()
        price_decision_data = {
         'expected_price': self.worker_expectations[1][0],
         'desired_price': self.desired_price,
         'real_price': self.avg_price,
         'consumption': self.consumption,
         'desired_consumption': self.desired_consumption,
         'max_price': max_price
          }


        desired_price = update_worker_price_expectation(price_decision_data)

        self.desired_price = desired_price


        wage_decision_data = {
          'expected_wage': self.worker_expectations[0][1],
          'desired_wage': self.desired_wage,
          'real_wage': self.wage,
          'working_hours': self.working_hours,
          'optimal_working_hours': self.optimals[1],
          'min_wage': self.get_min_wage()
        }


        desired_wage = update_worker_wage_expectation(wage_decision_data)

        self.desired_wage = desired_wage

        self.consumption = 0
        self.consumption_check = 0
        self.price = 0

    def get_min_wage(self):
        min_wage = self.model.config.MINIMUM_WAGE
        return min_wage

    def update_skills(self):
        if self.total_working_hours > 0:
            self.skills += self.model.config.SKILL_GROWTH_RATE
        else:
            self.skills -= self.model.config.SKILL_DECAY_RATE if self.skills > 0 else 0

    def get_hired(self, employer, wage, hours):
        self.employers[employer] = {'hours': hours, 'wage': wage}
        if hours<0:
          print("hours", hours)
          breakpoint()
        self.total_working_hours += hours
        self.update_average_wage()

    def update_hours(self, employer, hours):
        if hours < 0:
            print("hours", hours)
            breakpoint()
        if employer in self.employers:
            old_hours = self.employers[employer]['hours']
            self.employers[employer]['hours'] = hours
            self.total_working_hours += hours - old_hours
            self.update_average_wage()

    def get_fired(self, employer, layoff=False):
        if employer in self.employers:
            self.total_working_hours -= self.employers[employer]['hours']
            self.total_working_hours = max(0, self.total_working_hours)
            if not layoff:
              # When the worker quits the employer needs to be updated. If the worker is laid off, the employer is already updated
              employer.remove_worker(self)
            del self.employers[employer]
            self.update_average_wage()

    def quit(self, hours_to_quit):
        # Sort employers by hours worked, in descending order
        sorted_employers = sorted(self.employers.items(), key=lambda x: x[1]['hours'], reverse=True)

        for employer, details in sorted_employers:
            if hours_to_quit <= 0:
                break

            match hours_to_quit, details['hours']:
                case x, y if x >= y:
                    self.get_fired(employer)
                    hours_to_quit -= y
                case x, y if x < y:
                    self.update_hours(employer, y - x)
                    hours_to_quit = 0

        self.update_average_wage()


    def update_average_wage(self):
        if self.total_working_hours > 0:
                self.wage = sum(emp['wage'] * emp['hours'] for emp in self.employers.values()) / self.total_working_hours
                if self.wage < 0:
                  print("Worker attributes:")
                  for attr, value in self.__dict__.items():
                      print(f"{attr}: {value}")
                  breakpoint()

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
       max = self.savings/(self.model.time_horizon) + (self.wage * self.working_hours) if self.wage > 0 else  self.savings/(self.model.time_horizon)
       if max < self.savings:
         return max
       else:
         return self.savings


    def get_paid(self, wage):
        self.savings += wage

    def available_hours(self):

        if self.total_working_hours >= self.max_working_hours:
          if (self.total_working_hours- self.max_working_hours) > 1:
            print("Worker attributes:")
            for attr, value in self.__dict__.items():
                print(f"{attr}: {value}")
            breakpoint()
            return 0
        elif self.working_hours > self.total_working_hours:
            return self.working_hours - self.total_working_hours
        else:
          self.quit(self.total_working_hours - self.working_hours)
          return 0
