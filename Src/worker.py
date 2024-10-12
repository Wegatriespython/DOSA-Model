from sys import breakpointhook
from mesa import Agent
import numpy as np
from Utilities.Config import Config
from Utilities.utility_function import maximize_utility
from expectations import  get_market_demand, get_supply
from Utilities.adaptive_expectations import adaptive_expectations
from Utilities.Strategic_adjustments import update_worker_price_expectation, update_worker_wage_expectation
from Utilities.tools import update_dictionary

class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.employers = {}  # Dictionary to store employers and corresponding working hours
        self.total_working_hours = 0
        self.preference_mode = self.model.config.PREFERNCE_MODE_LABOR
        self.max_working_hours = 16
        self.worker_expectations = []
        self.skillscarbon = 1
        self.p_round_buyer = 0
        self.p_market_advantage = ""
        self.p_round_seller = 0
        self.consumption_check = 0
        self.desired_price = self.model.config.INITIAL_PRICE
        self.desired_wage = self.model.config.MINIMUM_WAGE
        self.dissatistifaction = 0
        self.wage_history1 = []
        self.price_history1 = []
        self.quantity_history1 = []
        self.supply_history1 = []
        self.avg_price = 0
        self.a_round_seller = 0
        self.market_advantage_seller = 0
        self.a_round_buyer = 0
        self.market_advantage_buyer = 0
        self.wage = model.config.MINIMUM_WAGE
        self.income = 0
        self.savings = self.model.config.INITIAL_SAVINGS
        self.optimals = []
        self.got_paid = False
        self.working_hours = 0
        self.leisure = 16
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
        self.worker_expectations = {
            'demand': {
                'labor': [model.config.INITIAL_LABOR_DEMAND] * model.time_horizon,
                'consumption': [model.config.INITIAL_CONSUMPTION_DEMAND] * model.time_horizon
            },
            'price': {
                'labor': [model.config.INITIAL_WAGE] * model.time_horizon,
                'consumption': [model.config.INITIAL_PRICE] * model.time_horizon
            },
            'supply': {
                'labor': [model.config.INITIAL_LABOR_SUPPLY] * model.time_horizon,
                'consumption': [model.config.INITIAL_CONSUMPTION_SUPPLY] * model.time_horizon
            }
        }
        self.demand_record = {'labor': [], 'consumption': []}
        self.price_record = {'labor': [], 'consumption': []}
        self.supply_record = {'labor': [], 'consumption': []}

        # Initialize these for immediate use
        self.expected_wage = model.config.INITIAL_WAGE
        self.expected_price = model.config.INITIAL_PRICE

    def step(self):
        #self.worker_expectations.clear()
        self.desired_consumption = 0
        self.desired_savings = 0
        self.update_expectations()
        self.update_utility()
        self.update_strategy()
        self.update_skills()

    def update_expectations(self):
        # Grab the demand for relevant goods
        labor_demand, labor_price, self.p_round_seller, self.p_market_advantage  = get_market_demand(self, 'labor')
        consumption_demand, consumption_price, self.p_round_buyer, _ = get_market_demand(self, 'consumption')
        labor_supply = get_supply(self, 'labor')
        consumption_supply = get_supply(self, 'consumption')

        demand = {
            'labor': labor_demand,
            'consumption': consumption_demand
        }
        price = {
            'labor': labor_price,
            'consumption': consumption_price
        }
        supply = {
            'labor': labor_supply,
            'consumption': consumption_supply
        }

        self.price_record = update_dictionary(price, self.price_record)
        self.demand_record = update_dictionary(demand, self.demand_record)
        self.supply_record = update_dictionary(supply, self.supply_record)

        if self.model.step_count < 2:
            return

        for category in ['demand', 'price', 'supply']:
            historical_data = (
                self.demand_record if category == 'demand' else
                self.price_record if category == 'price' else
                self.supply_record
            )
            self.worker_expectations[category] = adaptive_expectations(
                historical_data,
                self.worker_expectations[category],
                self.model.time_horizon
            )

        # For practical use only the future time horizon is needed
        self.worker_expectations = {
            category: {
                key: values[-self.model.time_horizon:]
                for key, values in self.worker_expectations[category].items()
            }
            for category in ['demand', 'price', 'supply']
        }

        # Update wage and price for immediate use
        self.expected_wage = self.worker_expectations['price']['labor'][0]
        self.expected_price = self.worker_expectations['price']['consumption'][0]

        return self.worker_expectations

    def update_utility(self):
        Utility_params = {
            'savings': round(self.savings, 2),
            'wage': self.worker_expectations['price']['labor'],
            'price': self.worker_expectations['price']['consumption'],
            'discount_rate': self.model.config.DISCOUNT_RATE,
            'time_horizon': self.model.time_horizon,
            'alpha': 0.9,
            'max_working_hours': 16,
            'working_hours': self.working_hours,            
            'expected_labor_demand': self.worker_expectations['demand']['labor'],
            'expected_consumption_supply': self.worker_expectations['supply']['consumption']
        }
        print(Utility_params)
        
        results = maximize_utility(Utility_params)
        self.desired_consumption, self.working_hours, self.leisure, self.desired_savings = [arr[0] for arr in results]
        working_hours_ratio = self.total_working_hours / self.working_hours if self.working_hours > 0 else 0
        # self.desired_consumption = self.desired_consumption * working_hours_ratio if working_hours_ratio > 0 and working_hours_ratio < 1 else self.desired_consumption
        self.optimals = [self.desired_consumption, self.working_hours, self.desired_savings]
        print(self.optimals)

    def update_strategy(self):
        max_price = self.get_max_consumption_price()
        price_decision_data = {
            'expected_price': self.worker_expectations['price']['consumption'][0],
            'desired_price': self.desired_price,
            'real_price': self.avg_price,
            'consumption': self.consumption,
            'desired_consumption': self.desired_consumption,
            'max_price': max_price,
            'a_round': self.a_round_buyer,
            'market_advantage': self.market_advantage_buyer
        }

        desired_price = update_worker_price_expectation(price_decision_data)
        self.desired_price = desired_price

        wage_decision_data = {
            'expected_wage': self.worker_expectations['price']['labor'][0],
            'desired_wage': self.desired_wage,
            'real_wage': self.wage,
            'working_hours': self.working_hours,
            'desired_working_hours': self.optimals[1],
            'min_wage': self.get_min_wage(),
            'a_round': self.a_round_seller,
            'market_advantage': self.market_advantage_seller
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

    def get_hired(self, employer, wage, hours, a_round = None, market_advantage = None):
        self.a_round_seller = a_round
        self.market_advantage_seller = market_advantage
        self.employers[employer] = {'hours': hours, 'wage': wage}
        if hours<0:
          print("hours", hours)
          breakpoint()
        self.total_working_hours += hours
        self.update_average_wage()

    def update_hours(self, employer, hours, a_round = None, market_advantage = None):
        self.a_round_seller = a_round
        self.market_advantage_seller = market_advantage
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
                    self.get_fired(employer, layoff=False)
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


    def consume(self, quantity, price, a_round, market_advantage):
        self.a_round_buyer = a_round
        self.market_advantage_buyer = market_advantage
        total_cost = quantity * price
        self.price_history.append(price)
        self.consumption += quantity
        self.consumption_check += quantity
        self.price = price
        self.prices.append(price)

        self.savings -= total_cost
        self.savings = max(0, self.savings) #prevent negative savings

    def get_max_consumption_price(self):
       max_price = self.savings/(self.model.time_horizon) + (self.wage * self.working_hours) if self.wage > 0 else  self.savings/(self.model.time_horizon)
       #max_price = self.worker_expectations['price']['consumption'][0] * 1.1
       #willing to pay up to 10% more than expected price
       if max_price < self.savings:
         return max_price
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