from mesa import Agent
import numpy as np
from Utilities.Simpler_profit_maxxin import profit_maximization
from Utilities.expectations import get_market_demand,get_supply
from Utilities.adaptive_expectations import adaptive_expectations
from Utilities.Strategic_adjustments import get_max_wage, get_min_sale_price, get_max_capital_price,calculate_production_capacity, get_desired_wage, get_desired_capital_price, get_desired_price, calculate_new_price
import logging
from math import isnan, nan


def update_dictionary(dictionary, dictionary_log):
  for key, value in dictionary.items():
    dictionary_log[key].append(value)
  return
def calculate_averages(dictionary, n=5):
    return {key: np.mean(value[-n:]) for key, value in dictionary.items()}
def dict_arithmetic(d1, d2, operation):
    return {key: operation(d1.get(key, 0), d2.get(key, 0))
            for key in set(d1) | set(d2)}

logging.basicConfig(level=logging.INFO)
class Firm(Agent):
    def __init__(self, unique_id, model, initial_capital, initial_productivity):
        super().__init__(unique_id, model)

    def update_firm_state(self):

        self.depreciation()


        self.sales = 0
        self.production = 0
        self.labor_demand = 0
        self.investment_demand = 0
        self.expected_demand = np.full(self.model.time_horizon, 0)
        self.expected_price = np.full(self.model.time_horizon, 0)
        self.productivity += round(self.labor_productivity,2)
        self.labor_productivity = 0
        labor = self.get_total_labor_units()
        if labor == 0:
          self.wage = self.model.config.INITIAL_WAGE
        else:
          self.pay_wages()
        self.update_average_price()
        ## Bugs in the employment and wage calculation, Sometimes firms get an inflated wage value with 0 employees.

        self.prices = []

    def depreciation(self):
      #Capital Depreciation
      capital = self.captal * self.model.config.DEPRECIATION_RATE
      self.capital = max(0, self.capital - capital)
      #Inventory Holding Costs
      inventory_cost = self.inventory * self.model.config.HOLDING_COST
      self.budget = max(0, self.budget - inventory_cost)
      return

    def performance_review(self):

        #Bottle Necks
        performance_actuals ={
        "production" :  self.production,
        "labor" : self.get_total_labor_units(),
        "capital" : self.capital,
        "price" : self.price,
        "sales" : self.sales_same_period,
        "profit" :  self.profit,
        "debt" :  self.debt}

        gaps = dict_arithmetic(self.optimals, performance_actuals, lambda x, y: x - y)

        match self.expectations['demand'], (self.optimals['sales'] - gaps['production']):
            case x, y if y > 0 and x >= 0:
                market_share = x / y
            case _, y if y == 0:
                market_share = 1
            case _:
                market_share = 0

        performance_actuals['market_share'] = market_share

        self.performance_record = update_dictionary(performance_actuals, self.performance_record)
        self.average_performance = calculate_averages(self.performance_record)
        self.gaps_record = update_dictionary(gaps, self.gaps_record)
        self.average_gaps = calculate_averages(self.gaps_record)

        return performance_actuals, gaps

    def update_expectations(self):
        # Grab the demand for relavent goods
        consumption_demand, consumption_price = get_market_demand(self, 'consumption')
        capital_demand, capital_price = get_market_demand(self, 'capital')
        labor_demand, wage = get_market_demand(self, 'labor')
        capital_supply = get_supply(self, 'capital')
        consumption_supply = get_supply(self, 'consumption')
        labor_supply = get_supply(self, 'labor')

        Demand ={
          'consumption': consumption_demand,
          'capital': capital_demand,
          'labor': labor_demand
        }
        Price = {
          'consumption': consumption_price,
          'capital': capital_price,
          'labor': wage
        }
        Supply = {
          'consumption': consumption_supply,
          'capital': capital_supply,
          'labor': labor_supply
        }

        self.price_record = update_dictionary(self.price_record, Price)
        self.demand_record = update_dictionary(self.demand_record, Demand)
        self.supply_record = update_dictionary(self.supply_record, Supply)
        #Maintain full period( history + time_horizon) forecasts, for use in adaptive expectations
        self.demand_expectations = adaptive_expectations(self.demand_record, self.demand_expectations, self.mode.time_horizon)
        self.price_expectations  = adaptive_expectations(self.price_record, self.price_expectations, self.mode.time_horizon)
        self.supply_expectations = adaptive_expectations(self.supply_record, self.supply_expectations, self.mode.time_horizon)
        #For practical use only the future time horizon is needed
        self.expectations = {
          'demand': self.demand_expectations[-self.model.time_horizon:],
          'price': self.price_expectations[-self.model.time_horizon:],
          'supply': self.supply_expectations[-self.model.time_horizon:]
        }
        return self.expectations

    def adjust_labor(self):
        optimal_labor = self.optimals['labor']
        self.labor_demand = max(0, optimal_labor - self.get_total_labor_units()) * self.max_working_hours  # Convert to hours
        if optimal_labor < self.get_total_labor_units():
            self.layoff_employees(self.get_total_labor_units() - optimal_labor)
        return self.labor_demand

    def adjust_production(self):
        optimal_production = self.optimals['production']
        self.production =  min(optimal_production, calculate_production_capacity(self.productivity, self.capital, self.capital_elasticity, self.get_total_labor_units()))
        self.inventory += max(0, self.production)
        return self.production

    def hire_worker(self, worker, wage, hours):
        if hours < 0:
          print("Negative hours")
          breakpoint()
        if worker in self.workers:
            #print("Worker already hired incresing hours")
            self.update_worker_hours(worker, hours)
        else:
            #print("Hiring new Worker")
            self.workers[worker] = {'hours':hours, 'wage':wage}
            self.total_working_hours += hours
            worker.get_hired(self, wage, hours)

    def update_worker_hours(self, worker, hours):
        if hours < 0:
          print("Negative hours")
          breakpoint()
        if worker in self.workers:
            self.workers[worker]['hours'] += hours
            self.total_working_hours += hours
            worker.update_hours(self, hours)

    def wage_adjustment(self):
      #Implement periodic wage adjustments, ideally workforce should quit to create turnover, causing firms to implement wage_adjustments to lower turnover, but non-essential for thesis.
      return


    def update_average_price(self):
        if len(self.prices) > 0 :
          self.prices = [p for p in self.prices if not np.isnan(p)]
          average_price = np.mean(self.prices)
          self.price = average_price




    def pay_wages(self):
      # Fire workers, if their skill adjusted wage is higher than market avg.

        fire_list = []
        wage_total = 0
        employees_total = len(self.workers)
        skill_total = 0
        self.wage_adjustment()
        for worker in self.workers:
            wage = self.workers[worker]['wage']
            hours = self.workers[worker]['hours']
            if hours < 0:
              print("Negative hours")
              breakpoint()
            wage_total += wage
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
            self.labor_productivity = skill_total/employees_total
            print("skill total", skill_total)
            #this would be cumilative over periods. The correct way would be a static effect which grows only if labor_productivity rises over time.:
        return self.wage

    def layoff_employees(self, units_to_fire):
        hours_to_fire = units_to_fire * self.max_working_hours
        hours_fired = 0
        fire_list = []
        # Sort workers by hours worked to fire the least productive workers first
        self.workers = dict(sorted(self.workers.items(), key=lambda item: item[1]['hours']))

        for worker in self.workers:
            hours = self.workers[worker]['hours']
            if hours<0:
              print("Negative hours")
              breakpoint()
            if hours_fired + hours <= hours_to_fire:
                fire_list.append(worker)
                hours_fired += hours
            else:
                break
        for worker in fire_list:
            self.fire_worker(worker)
        #calculate new average wage
        self.calculate_average_wage()

    def fire_worker(self, worker):
        if worker in self.workers:
            hours = self.workers[worker]['hours']
            if hours < 0:
              print("Negative hours")
              breakpoint()
            self.total_working_hours -= hours
            del self.workers[worker]
            worker.get_fired(self, layoff=True)

    # To be called only when workers quit, not along w firing.
    def remove_worker(self, worker):
          hours = self.workers[worker]['hours']
          self.total_working_hours -= hours
          del self.workers[worker]



    def calculate_average_wage(self):
        if self.workers == {}:
            wage_avg = self.model.data_collector.get_average_wage(self.model)
            wage_avg = max(wage_avg, self.model.config.MINIMUM_WAGE)
            return wage_avg
        else:
            total_wage_payout = sum(self.workers[worker]['wage'] * self.workers[worker]['hours'] for worker in self.workers)
            total_hours = sum(self.workers[worker]['hours'] for worker in self.workers)
            if total_hours < 0:
              print("Negative hours")
              breakpoint()
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
            #optimal_debt = self.optimals['debt']
           # if optimal_debt - self.debt  <= 0:
              #if not taking debt, purely finance out of budget.
            self.budget -= budget_change
            #else:
              #if optimal_debt<= budget_change:
                #self.budget -= budget_change - optimal_debt
                #self.debt += optimal_debt
                #else:
                #self.debt += budget_change
    def sell_goods(self, quantity, price):
        inventory_change = self.inventory - quantity

        self.inventory -= quantity
        self.sales += quantity
        budget_change = quantity * price
        self.budget += quantity * price  # Price already adjusted in execute_capital_market
        self.prices.append(price)
        self.sales_same_period += quantity


    def nash_improvements(self):

      match self.firm_type, self.model.step_count:
        #first 2 periods use starting values
        case _, y if y < 2:
          self.desireds = {
            'wage': self.wage,
            'price': self.price,
            'capital_price': self.model.config.INITIAL_RELATIVE_PRICE_CAPITAL
          }
          return
        #for consumption firms
        case 'consumption', _:
          price_expectations = self.expectations['price']['consumption']
          desired_capital_price = get_desired_capital_price(self)
        #for capital firms
        case 'capital', _:
          price_expectations = self.expectations['price']['capital']
          desired_capital_price = 0
        # Doesn't happen but just for pyright to not show red squiggly lines
        case _,_:
          print('invalid firm type')
          breakpoint()
          return

      wage_expectations = self.expectations['price']['labor']

      desired_price = calculate_new_price(price_expectations, self.desireds['price'],self.price, self.sales, self.optimals['sales'],  self.zero_profit_conditions['price'], self.optimals['inventory'], self.inventory, self.production_gap)

      desired_wage = calculate_new_price(wage_expectations,self.desireds['price'],self.wage, self.optimals['labor'], self.get_total_labor_units(), self.zero_profit_conditions['wage'], self.model.config.MINIMUM_WAGE)

      self.desireds = {
        'wage': desired_wage,
        'price': desired_price,
        'capital_price': desired_capital_price}

      self.desireds_record = update_dictionary(self.desireds, self.desireds_record)

      return self.desireds


    def get_zero_profit_conditions(self):

      max_wage = get_max_wage(self.total_working_hours, self.productivity, self.capital, self.capital_elasticity, self.price, self.get_total_labor_units(),self.optimals['labor'],self.model.config.MINIMUM_WAGE)
      min_sale_price = get_min_sale_price(self.firm_type, self.workers, self.productivity, self.capital, self.capital_elasticity, self.get_total_labor_units(), self.inventory)

      if min_sale_price < 0.5:
        breakpoint()

      max_capital_price = get_max_capital_price(self.investment_demand, self.optimals['production'],self.optimals['capital'], self.price, self.capital_elasticity, self.model.time_horizon, self.model.config.DISCOUNT_RATE)
      self.zero_profit_conditions = {
      'wage':max_wage,
      'price': min_sale_price,
      'capital_price': max_capital_price}

      return self.zero_profit_conditions




class Firm2(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM2_INITIAL_CAPITAL, model.config.INITIAL_PRODUCTIVITY)
        self.capital_inventory = 0  # Separate inventory for capital goods
        self.firm_type = 'consumption'
        self.capital_resale_price = 0
        self.labor_productivity = 0
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
        self.expected_demand = [model.config.FIRM2_INITIAL_DEMAND]*self.model.time_horizon
        self.expected_price = [self.price]*self.model.time_horizon
        self.quality = 1
        self.carbon_intensity = 1
        self.preference_mode = self.model.config.PREFERNCE_MODE_CONSUMPTION


    def adjust_investment_demand(self):
          optimal_capital = self.optimals["capital"]
          self.investment_demand = self.optimals["investment"]
          if optimal_capital < self.capital:
              self.capital_inventory = 0

              self.capital_resale_price = self.model.data_collector.get_average_capital_price(self.model)
              self.captial_min_price = 0.1

          return self.investment_demand





    def make_production_decision(self):
        average_capital_price = self.model.data_collector.get_average_capital_price(self.model)
        number_of_firms = sum(1 for agent in self.model.schedule.agents if isinstance(agent, (Firm2)))
        Profit_max_params = {
            'current_capital': round(self.capital,2),
            'current_labor': round(self.total_labor_units,2),
            'current_price': round(self.price,2),
            'productivity': self.productivity,
            'expected_demand': self.expected_demand/number_of_firms,
            'expected_price': self.expected_price,
            'capital_price': self.expectations[2],
            'capital_elasticity': self.capital_elasticity,
            'inventory': round(self.inventory,2),
            'depreciation_rate': self.model.config.DEPRECIATION_RATE,
            'time_horizon': (self.model.time_horizon),
            'discount_rate': self.model.config.DISCOUNT_RATE,
            'budget': round(self.budget,2),
            'wage': round(self.wage * self.max_working_hours,2),
            'expected_capital_supply': self.expectations[3],
            'expected_labor_supply': self.expectations[4],
            'debt': self.debt,
            'carbon_intensity': self.carbon_intensity,
            'new_capital_carbon_intensity': 1,
            'carbon_tax_rate': 0,
            'holding_costs': self.model.config.HOLDING_COST
        }
        print("Calling profit_maximization with parameters:" , Profit_max_params)
        result = profit_maximization(Profit_max_params)

        if result is not None:
          optimal_labor = result['optimal_labor']
          optimal_capital = result['optimal_capital']
          optimal_production = result['optimal_production']
          optimal_inventory = result['optimal_inventory']
          optimal_investment = result['optimal_investment']
          optimal_sales = result['optimal_sales']
          optimal_debt = result['optimal_debt']
          optimal_interest_payment = result['optimal_interest_payment']
          optimal_debt_payment = result['optimal_debt_payment']
          optimal_carbon_intensity = result['optimal_carbon_intensity']
          optimal_emissions = result['optimal_emissions']
          optimal_carbon_tax_payments = result['optimal_carbon_tax_payment']

        self.optimals = {
            'labor': optimal_labor[0],
            'capital': optimal_capital[0],
            'production': optimal_production[0],
            'inventory': optimal_inventory[0],
            'sales': optimal_sales[0],
            'debt': optimal_debt[0],
            'debt_payment': optimal_debt_payment[0],
            'investment': optimal_investment[0]
        }
        print(f"Optimal values: {self.optimals}")

        """self.zero_profit_conditions = {
            'wage': zero_profit_wage,
            'price': zero_profit_price,
            'capital_price': zero_profit_capital_price}"""
        """if zero_profit_conditions is not None:
          zero_profit_wages = zero_profit_conditions['wage']
          zero_profit_wage = zero_profit_wages[0]
          zero_profit_prices = zero_profit_conditions['price']
          zero_profit_price = zero_profit_prices[0]
          zero_profit_capital_prices = zero_profit_conditions['capital_price']
          print("Zero profit capital prices", zero_profit_capital_prices)
          zero_profit_capital_price = zero_profit_capital_prices[0]"""
        """zero_profit_wage = self.wage
        zero_profit_price = self.price
        zero_profit_capital_price = self.expectations[2]"""

        return self.optimals
