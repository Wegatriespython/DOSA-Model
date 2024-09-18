from mesa import Agent
import numpy as np
from Utilities.Simpler_profit_maxxin import profit_maximization
from Utilities.expectations import expect_demand, expect_price, get_market_demand, get_expectations,get_supply, expect_price_ar
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
        self.sales_same_period =0
        self.debt = 0
        self.time_horizon = self.model.config.TIME_HORIZON - self.model.step_count
        self.total_labor_units = 0
        self.historic_labor_prices= []
        self.per_worker_income = 0
        self.market_share = 0
        self.market_share_history = []
        self.zero_profit_conditions = []
        self.zero_profit_conditions_cache = []
        self.historic_demand = []
        self.historic_price = []
        self.optimals = {}
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


        self.historic_sales.append(self.sales_same_period)
        self.sales_same_period = 0
        sales_avg = np.mean(self.historic_sales[-5:])
        market_share = sales_avg / self.optimals['sales'] if self.optimals else 1
        self.market_share_history.append(self.market_share)
        if self.model.step_count > 5:
          self.market_share = market_share
          if self.market_share > .5:
            print("sales_avg", sales_avg)
            print("market share" , self.market_share)
            breakpoint()
        self.sales = 0
        self.production = 0
        self.labor_demand = 0
        self.investment_demand = 0
        self.expected_demand = np.full(self.time_horizon, 0)
        self.expected_price = np.full(self.time_horizon, 0)
        self.productivity += round(self.labor_productivity,2)
        self.labor_productivity = 0
        self.get_total_labor_units()
        self.pay_wages()
        self.update_average_price()
        self.prices = []
        if self.budget < self.wage:
            self.model.schedule.remove(self)


    def update_expectations(self):

        if self.model.step_count < 1:
            self.expected_demand= np.full(self.time_horizon ,30)
            self.expected_price = np.full(self.time_horizon,1)
            self.expectations =[np.mean(self.expected_demand), np.mean(self.expected_price), 3, 5, 96, 0.0625]
            self.desireds = [self.wage, self.price, self.model.config.INITIAL_RELATIVE_PRICE_CAPITAL]

            return
        price_capital = 0
        demand, price =  get_market_demand(self, self.firm_type)
        if self.firm_type == 'consumption':
          capital_demand, price_capital = get_market_demand(self, 'capital')
          print("capital price", np.mean(price_capital))

        self.historic_demand.append(demand)
        self.historic_price.append(price)

        labor_demand, wage = get_market_demand(self,'labor')
        self.historic_labor_prices.append(wage)
        expected_capital_supply = get_supply(self, "capital")
        expected_labor_supply = get_supply(self, "labor")

        self.expected_price, self.expected_demand = get_expectations(demand, self.historic_demand,  price ,self.historic_price,(self.time_horizon))


        expected_wages = expect_price_ar(self.historic_labor_prices, wage, self.time_horizon)
        self.expectations = [self.expected_demand[0], self.expected_price[0], price_capital, expected_capital_supply, expected_labor_supply, expected_wages[0]]
        self.expectations_cache.append(self.expectations)
        return



    def make_production_decision(self):


        self.historic_sales.append(self.sales)
        if len(self.historic_sales)>5:
            self.historic_sales = self.historic_sales[-5:]


        average_capital_price = self.model.data_collector.get_average_capital_price(self.model)

        if self.budget <= 0:

            return # Skip production if budget is negative
        print("Calling profit_maximization with parameters:", {
            'current capital': round(self.capital,2),
            'current labor': round(self.total_labor_units,2),
            'current price': round(self.price,2),
            'productivity': self.productivity,
            'expected demand': self.expected_demand * self.market_share,
            'expected price': self.expected_price,
            'expectated capital price': self.expectations[2],
            'capital elasticity': self.capital_elasticity,
            'inventory': round(self.inventory,2),
            'depreciation rate': self.model.config.DEPRECIATION_RATE,
            'time horizon': (self.time_horizon),
            'discount rate': self.model.config.DISCOUNT_RATE,
            'budget': round(self.budget,2),
            'wage': round(self.wage * self.max_working_hours,2),
            'expected capital supply': self.expectations[3],
            'expected labor supply': self.expectations[4],
            'debt': self.debt,
            'carbon intensity': self.carbon_intensity,
            'new capital carbon intensity': 1,
            'carbon_tax_rate': 0
        })

        self.per_worker_income = self.wage * self.max_working_hours

        result = profit_maximization(
            round(self.capital,2),
            round(self.total_labor_units,2),
            round(self.price,2),
            self.productivity,
            self.expected_demand,
            self.expected_price,
            self.expectations[2],  # Updated
            self.capital_elasticity,
            round(self.inventory,2),
            self.model.config.DEPRECIATION_RATE,
            (self.time_horizon),
            self.model.config.DISCOUNT_RATE,
            round(self.budget,2),
            round(self.wage * self.max_working_hours,2),
            self.expectations[3], #Capital Supply,
            self.expectations[4],
            self.debt,
            self.carbon_intensity,
            1,
            0#Labor Supply,
            )

        if result is None:
            print("Optimization failed")
            return


        optimal_labor = result['optimal_labor']
        optimal_capital = result['optimal_capital']
        optimal_production = result['optimal_production']
        optimal_inventory = result['optimal_inventory']
        optimal_investment = result['optimal_investment']
        optimal_sales = result['optimal_sales']
        optimal_debt = result['optimal_debt']
        optimal_debt_payment = result['optimal_debt_payment']
        optimal_profit_per_period = result['profits_per_period']
        optimal_carbon_intensity = result['optimal_carbon_intensity']
        optimal_emissions = result['optimal_emissions']
        optimal_carbon_tax_payments = result['optimal_carbon_tax_payments']

        self.optimals = {
            'labor': optimal_labor[0],
            'capital': optimal_capital[0],
            'production': optimal_production[0],
            'inventory': optimal_inventory[0],
            'sales': optimal_sales[0],
            'debt': optimal_debt[0],
            'debt_payment': optimal_debt_payment[0],
            'profit_per_period': optimal_profit_per_period[0],
            'investment': optimal_investment[0]
        }
        print(f"Optimal values: {self.optimals}")




        return optimal_labor, optimal_capital, optimal_production

    def adjust_labor(self):
        if self.inventory > self.model.config.INVENTORY_THRESHOLD:
            self.labor_demand = 0
            return self.labor_demand
        optimal_labor = self.optimals['labor']

        self.labor_demand = max(0, optimal_labor - self.get_total_labor_units()) * self.max_working_hours  # Convert to hours

        if optimal_labor < self.get_total_labor_units():
            self.layoff_employees(self.get_total_labor_units() - optimal_labor)
        return self.labor_demand
    def adjust_investment_demand(self):
        if self.firm_type == 'consumption':
            if self.inventory > self.model.config.INVENTORY_THRESHOLD:
                self.investment_demand = 0
                return self.investment_demand
            optimal_capital = self.optimals["capital"]
            self.investment_demand = self.optimals["investment"]
            """if optimal_capital < self.capital:
                self.capital_inventory = 0

                self.capital_resale_price = self.model.data_collector.get_average_capital_price(self.model)
                self.captial_min_price = 0.1"""
            return self.investment_demand
    def adjust_production(self):
        if self.inventory > self.model.config.INVENTORY_THRESHOLD:
            self.production = 0
            return self.production
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
      if self.model.step_count >2:

        if len(self.prices) > 0 :
          self.prices = [p for p in self.prices if not np.isnan(p)]
          average_price = np.mean(self.prices)

          self.price = average_price
        else :
          # no transactions have been made
          self.price = 0



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
            optimal_debt = self.optimals['debt']
            if optimal_debt - self.debt  <= 0:
              #if not taking debt, purely finance out of budget.
              self.budget -= budget_change
            else:
              if optimal_debt<= budget_change:
                self.budget -= budget_change - optimal_debt
                self.debt += optimal_debt
              else:
                self.debt += budget_change



    def sell_goods(self, quantity, price):
        inventory_change = self.inventory - quantity

        self.inventory -= quantity
        self.sales += quantity
        budget_change = quantity * price
        self.budget += quantity * price  # Price already adjusted in execute_capital_market
        self.prices.append(price)
        self.sales_same_period += quantity


    def nash_improvements(self):

      if self.model.step_count < 2:
        self.desireds = [self.wage, self.price, self.model.config.INITIAL_RELATIVE_PRICE_CAPITAL]
        return
      if self.firm_type == 'consumption':
            desired_price = get_desired_price(self.expectations[1], self.desireds[1],self.price,self.sales, self.optimals['sales'],  self.zero_profit_conditions[1], self.optimals['inventory'], self.inventory)

            desired_capital_price = get_desired_capital_price(self)
      else:
            desired_price = get_desired_price(self.expectations[1], self.desireds[1],self.price,self.sales, self.optimals['sales'],  self.zero_profit_conditions[1], self.optimals['inventory'], self.inventory)
            desired_capital_price = 0
      desired_wage = get_desired_wage(self.expectations[5],self.desireds[0],self.wage, self.optimals['labor'], self.get_total_labor_units(), self.zero_profit_conditions[0], self.model.config.MINIMUM_WAGE)

      self.desireds = [desired_wage, desired_price, desired_capital_price]



      return self.desireds

    def get_zero_profit_conditions(self):

      max_wage = get_max_wage(self.total_working_hours, self.productivity, self.capital, self.capital_elasticity, self.price, self.get_total_labor_units(),self.optimals['labor'],self.model.config.MINIMUM_WAGE)
      min_sale_price = get_min_sale_price(self.firm_type, self.workers, self.productivity, self.capital, self.capital_elasticity, self.get_total_labor_units(), self.inventory)

      if min_sale_price < 0.5:
        breakpoint()

      max_capital_price = get_max_capital_price(self.investment_demand, self.optimals['production'],self.optimals['capital'], self.price, self.capital_elasticity, self.time_horizon, self.model.config.DISCOUNT_RATE)
      self.zero_profit_conditions = [max_wage, min_sale_price, max_capital_price]
      self.zero_profit_conditions_cache.append(self.zero_profit_conditions)
      return self.zero_profit_conditions



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
        self.expected_demand = [model.config.FIRM1_INITIAL_DEMAND] *self.time_horizon
        self.expected_price = [self.price] * self.time_horizon
        self.productivity = model.config.INITIAL_PRODUCTIVITY
        self.carbon_intensity = 1
        self.labor_productivity = 0
        self.preference_mode = self.model.config.PREFERNCE_MODE_CAPITAL

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
        self.labor_productivity = 0
        self.capital_elasticity = model.config.CAPITAL_ELASTICITY_FIRM2
        self.investment_demand = model.config.FIRM2_INITIAL_INVESTMENT_DEMAND
        self.capital = model.config.FIRM2_INITIAL_CAPITAL
        self.productivity = model.config.INITIAL_PRODUCTIVITY
        self.inventory = model.config.FIRM2_INITIAL_INVENTORY
        self.historic_demand = [model.config.FIRM2_INITIAL_DEMAND]
        self.budget = 5
        self.carbon_intensity = 1
        self.historic_sales = [model.config.INITIAL_SALES]
        self.price = self.model.config.INITIAL_PRICE
        self.historic_inventory = [self.inventory]
        self.expected_demand = [model.config.FIRM2_INITIAL_DEMAND]*self.time_horizon
        self.expected_price = [self.price]*self.time_horizon
        self.quality = 1
        self.carbon_intensity = 1
        self.preference_mode = self.model.config.PREFERNCE_MODE_CONSUMPTION
