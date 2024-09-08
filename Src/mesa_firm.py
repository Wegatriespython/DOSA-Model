from mesa import Agent
import numpy as np
from Utilities.Simpler_profit_maxxin import profit_maximization
from Utilities.expectations import expect_demand, expect_price, get_market_demand, get_expectations,get_supply
from Utilities.Strategic_adjustments import get_max_wage, get_min_sale_price, get_max_capital_price,calculate_production_capacity, get_desired_wage, get_desired_capital_price, get_desired_price
import logging

logging.basicConfig(level=logging.INFO)
class Firm(Agent):
    def __init__(self, unique_id, model, initial_capital, initial_productivity):
        super().__init__(unique_id, model)
        self.workers = {}  # Dictionary to store workers and their working hours
        self.total_working_hours = 0
        self.prices = []
        self.desireds = []
        self.total_labor_units = 0
        self.per_worker_income = 0
        self.zero_profit_conditions = []
        self.zero_profit_conditions_cache = []
        self.historic_demand = []
        self.historic_price = []
        self.histiric_sales = []
        self.optimals = []
        self.optimals_cache = []
        self.expectations = []
        self.expectations_cache = []
        self.production = 0
        self.sales = 0
        self.firm_type = ''
        self.labor_demand = 0
        self.investment_demand = 0
        self.mode = 'decentralized'
        self.wage = model.config.INITIAL_WAGE
        self.max_working_hours = model.config.MAX_WORKING_HOURS

    def update_firm_state(self):
       #self.train_demand_predictor()
        depreciation_amount = max(0, self.inventory * self.model.config.DEPRECIATION_RATE)
        self.inventory = max(0, self.inventory - depreciation_amount)
        if self.firm_type == 'consumption':
            capital_depreciation = self.capital * self.model.config.DEPRECIATION_RATE
            self.capital = max(0, self.capital - capital_depreciation)
        self.sales = 0
        self.production = 0
        self.labor_demand = 0
        self.investment_demand = 0
        self.expected_demand = 0
        self.expected_price = 0
        self.get_total_labor_units()
        self.pay_wages()
        self.update_average_price()

    def update_expectations(self):

        if self.model.step_count < 1:
            self.expected_demand= np.full(self.model.config.TIME_HORIZON,6)
            self.expected_price = np.full(self.model.config.TIME_HORIZON,1)
            self.expectations =[np.mean(self.expected_demand), np.mean(self.expected_price)]
            return

        demand, price =  get_market_demand(self, self.firm_type)

        self.historic_demand.append(demand)
        self.historic_price.append(price)
        self.expected_demand, self.expected_price = get_expectations(self, self.historic_demand, self.historic_price, self.model.config.TIME_HORIZON)
        self.expectations = [np.mean(self.expected_demand), np.mean(self.expected_price)]
        self.expectations_cache.append(self.expectations)
        return



    def make_production_decision(self):
        self.historic_sales.append(self.sales)
        if len(self.historic_sales)>5:
            self.historic_sales = self.historic_sales[-5:]


        average_capital_price = self.model.data_collector.get_average_capital_price(self.model)

        if self.budget < 0:

            return # Skip production if budget is negative
        print("Calling profit_maximization with parameters:", {
            'current_capital': self.capital,
            'current_labor': self.total_labor_units,
            'current_price': self.price,
            'current_productivity': self.productivity,
            'expected_demand': self.expected_demand,
            'expected_price': self.expected_price,
            'capital_price': 3,  # Updated
            'capital_elasticity': self.capital_elasticity,
            'current_inventory': self.inventory,
            'depreciation_rate': self.model.config.DEPRECIATION_RATE,
            'expected_periods': (self.model.config.TIME_HORIZON),
            'discount_rate': self.model.config.DISCOUNT_RATE,
            'budget': self.budget,
            'wage': self.wage * self.max_working_hours # Per unit wage
        })
        self.per_worker_income = self.wage * self.max_working_hours

        result = profit_maximization(
            self.capital,
            self.total_labor_units,
            self.price,
            self.productivity,
            self.expected_demand,
            self.expected_price,
            3,  # Updated
            self.capital_elasticity,
            self.inventory,
            self.model.config.DEPRECIATION_RATE,
            (self.model.config.TIME_HORIZON),
            self.model.config.DISCOUNT_RATE,
            self.budget,
            self.wage * self.max_working_hours # Per unit wage
        )

        if result is None:
            print("Optimization failed")
            return


        optimal_labor = round(result['optimal_labor'], 1)
        optimal_capital = round(result['optimal_capital'], 1)
        optimal_production = round(result['optimal_production'],1)
        Trajectory_inventory = result['optimal_inventory'],1
        optimal_inventory = round(round(Trajectory_inventory[0][0],1),1)
        optimal_sales = result['optimal_sales']
        optimal_sales = round(np.mean(optimal_sales),1)


        self.optimals = [optimal_labor, optimal_capital, optimal_production, optimal_inventory, optimal_sales]
        print("Optimal values:", self.optimals)
        self.optimals_cache.append(self.optimals)
        # Keep only the last 5 values
        if len(self.optimals_cache) > 5:
            self.optimals_cache = self.optimals_cache[-5:]

        # Calculate the mean of the available optimal values
        self.optimals = np.mean(self.optimals_cache, axis=0)
        return optimal_labor, optimal_capital, optimal_production

    def adjust_labor(self):
        if self.inventory > self.model.config.INVENTORY_THRESHOLD:
            self.labor_demand = 0
            return self.labor_demand
        optimal_labor = self.optimals[0]

        self.labor_demand = max(0, optimal_labor - self.get_total_labor_units()) * self.max_working_hours  # Convert to hours

        if optimal_labor < self.get_total_labor_units():
            self.layoff_employees(self.get_total_labor_units() - optimal_labor)
        return self.labor_demand
    def adjust_investment_demand(self):
        if self.firm_type == 'consumption':
            if self.inventory > self.model.config.INVENTORY_THRESHOLD:
                self.investment_demand = 0
                return self.investment_demand
            optimal_capital = self.optimals[1]
            self.investment_demand = max(0, optimal_capital - self.capital)
            if optimal_capital < self.capital:
                self.capital_inventory = 0

                self.capital_resale_price = self.model.data_collector.get_average_capital_price(self.model)
                self.captial_min_price = 0.1
            return self.investment_demand
    def adjust_production(self):
        if self.inventory > self.model.config.INVENTORY_THRESHOLD:
            self.production = 0
            return self.production
        optimal_production = self.optimals[2]
        self.production =  min(optimal_production, calculate_production_capacity(self.productivity, self.capital, self.capital_elasticity, self.get_total_labor_units()))

        self.inventory += max(0, self.production)
        return self.production

    def hire_worker(self, worker, wage, hours):
        if worker in self.workers:
            #print("Worker already hired incresing hours")
            self.update_worker_hours(worker, hours)
        else:
            #print("Hiring new Worker")
            self.workers[worker] = {'hours':hours, 'wage':wage}
            self.total_working_hours += hours
            worker.get_hired(self, wage, hours)

    def update_worker_hours(self, worker, hours):
        if worker in self.workers:
            self.workers[worker]['hours'] += hours
            self.total_working_hours += hours
            worker.update_hours(self, hours)
    def wage_adjustment(self):
        for worker in self.workers:
            self.workers[worker]['wage'] = self.desireds[0]

    def update_average_price(self):
      if len(self.prices) > 5:
        self.prices = self.prices[-5:]
        average_price = np.mean(self.prices)
        self.price = average_price



    def pay_wages(self):
      # Fire workers, if their skill adjusted wage is higher than market avg.

        fire_list = []
        wage_total = 0
        employees_total=0
        skill_total = 0
        self.wage_adjustment()
        for worker in self.workers:
            wage = self.workers[worker]['wage']
            hours = self.workers[worker]['hours']
            wage_total += wage
            employees_total += 1
            skill_total += worker.skills
            budget_change = wage * hours

            if self.budget >= budget_change:

                self.budget -= budget_change

                worker.get_paid(budget_change)
            else:
                worker.got_paid = False
                fire_list.append(worker)

        for worker in fire_list:
            self.fire_worker(worker)
        if employees_total > 0:
            self.wage = wage_total/employees_total
        return self.wage




    def layoff_employees(self, units_to_fire):
        hours_to_fire = units_to_fire * self.max_working_hours
        hours_fired = 0
        fire_list = []
        # Sort workers by hours worked to fire the least productive workers first
        self.workers = dict(sorted(self.workers.items(), key=lambda item: item[1]['hours']))

        for worker in self.workers:
            hours = self.workers[worker]['hours']
            if hours_fired + hours <= hours_to_fire:
                fire_list.append(worker)
                hours_fired += hours
            else:
                break
        for worker in fire_list:
            self.fire_worker(worker)
    def fire_worker(self, worker):
        if worker in self.workers:
            hours = self.workers[worker]['hours']
            self.total_working_hours -= hours
            del self.workers[worker]
            worker.get_fired(self)
    def calculate_average_wage(self):
        if self.workers == {}:
            wage_avg = self.model.data_collector.get_average_wage(self.model)
            wage_avg = max(wage_avg, self.model.config.MINIMUM_WAGE)
            return wage_avg
        else:
            total_wage_payout = sum(self.workers[worker]['wage'] * self.workers[worker]['hours'] for worker in self.workers)
            total_hours = sum(self.workers[worker]['hours'] for worker in self.workers)
            wage_avg = total_wage_payout/ total_hours if total_hours > 0 else 0
            wage_avg = max(wage_avg, self.model.config.MINIMUM_WAGE)
            return wage_avg


    def get_total_labor_units(self):
        self.total_labor_units = self.total_working_hours / self.max_working_hours

        return self.total_labor_units
    def buy_capital(self, quantity, price):
        if isinstance(self, Firm2):
            self.capital += quantity
            self.investment_demand -= quantity

            budget_change = quantity * price

            self.budget -= budget_change

    def sell_goods(self, quantity, price):
        inventory_change = self.inventory - quantity

        self.inventory -= quantity
        self.sales += quantity
        budget_change = quantity * price
        self.budget += quantity * price  # Price already adjusted in execute_capital_market
        self.prices.append(price)
    def nash_improvements(self):

      if self.model.step_count < 2:
        self.desireds = [self.wage, self.price, self.model.config.INITIAL_RELATIVE_PRICE_CAPITAL]
        return
      labor_supply = get_supply(self,"labor")
      labor_demand, labor_price = get_market_demand(self,"labor")
      capital_supply = get_supply(self,"capital")
      capital_demand, capital_price = get_market_demand(self,"capital")
      consumption_supply = get_supply(self,"consumption")
      consumption_demand, consumption_price = get_market_demand(self,"consumption")
      actual_labor = self.get_total_labor_units()
      desired_wage = get_desired_wage(self, labor_supply, labor_demand, actual_labor)
      print(f"Desired Wage: {desired_wage}, labor_supply: {labor_supply}, labor_demand: {labor_demand}")
      if self.firm_type == 'consumption':
        desired_price = get_desired_price(self, consumption_supply, consumption_demand,self.desireds[1], self.zero_profit_conditions[1], self.price)
        print(f"Desired Price: {desired_price}, consumption_supply: {consumption_supply}, consumption_demand: {consumption_demand}")
        desired_capital_price = get_desired_capital_price(self)
      else:
        desired_price = get_desired_price(self,capital_supply, capital_demand, self.desireds[1], self.zero_profit_conditions[1], self.price)
        desired_capital_price = desired_price

      self.desireds = [desired_wage, desired_price, desired_capital_price]
      return self.desireds

    def get_zero_profit_conditions(self):
      max_wage = get_max_wage(self.total_working_hours, self.productivity, self.capital, self.capital_elasticity, self.price, self.get_total_labor_units(), self.optimals, self.model.config.MINIMUM_WAGE)
      min_sale_price = get_min_sale_price(self.firm_type, self.workers, self.productivity, self.capital, self.capital_elasticity, self.get_total_labor_units(), self.inventory)
      max_capital_price = get_max_capital_price(self.investment_demand, self.optimals, self.price, self.capital_elasticity, self.model.config.TIME_HORIZON, self.model.config.DISCOUNT_RATE)
      self.zero_profit_conditions = [max_wage, min_sale_price, max_capital_price]
      self.zero_profit_conditions_cache.append(self.zero_profit_conditions)
      return self.zero_profit_conditions


    def update_after_markets(self):
        pass


class Firm1(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM1_INITIAL_CAPITAL, model.config.INITIAL_PRODUCTIVITY)
        self.firm_type = 'capital'
        self.capital_elasticity = model.config.CAPITAL_ELASTICITY_FIRM1
        self.price = self.model.config.INITIAL_PRICE * self.model.config.INITIAL_RELATIVE_PRICE_CAPITAL
        self.capital = model.config.FIRM1_INITIAL_CAPITAL
        self.inventory = model.config.FIRM1_INITIAL_INVENTORY
        self.historic_demand = [model.config.FIRM1_INITIAL_DEMAND]
        self.budget = self.capital
        self.historic_sales = [model.config.INITIAL_SALES]
        self.historic_inventory = [self.inventory]
        self.expected_demand = model.config.FIRM1_INITIAL_DEMAND
        self.expected_price = self.price
    def step(self):
        super().step()
  # Reset price to initial value
        if self.mode == 'decentralized':
            self.innovate()

    def innovate(self):
        if self.model.random.random() < self.model.config.INNOVATION_ATTEMPT_PROBABILITY:
            rd_investment = self.capital * self.model.config.FIRM1_RD_INVESTMENT_RATE

            self.budget -= rd_investment
            if self.model.random.random() < self.model.config.PRODUCTIVITY_INCREASE_PROBABILITY:
                self.productivity *= (1 + self.model.config.PRODUCTIVITY_INCREASE)

class Firm2(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM2_INITIAL_CAPITAL, model.config.INITIAL_PRODUCTIVITY)
        self.capital_inventory = 0  # Separate inventory for capital goods
        self.firm_type = 'consumption'
        self.capital_resale_price = 0
        self.capital_elasticity = model.config.CAPITAL_ELASTICITY_FIRM2
        self.investment_demand = model.config.FIRM2_INITIAL_INVESTMENT_DEMAND
        self.capital = model.config.FIRM2_INITIAL_CAPITAL
        self.productivity = model.config.INITIAL_PRODUCTIVITY
        self.inventory = model.config.FIRM2_INITIAL_INVENTORY
        self.historic_demand = [model.config.FIRM2_INITIAL_DEMAND]
        self.budget = 5
        self.historic_sales = [model.config.INITIAL_SALES]
        self.price = self.model.config.INITIAL_PRICE
        self.historic_inventory = [self.inventory]
        self.expected_demand = model.config.FIRM2_INITIAL_DEMAND
        self.expected_price = self.price
