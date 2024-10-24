from mesa import Agent
import numpy as np
from Utilities.Simpler_profit_maxxin import profit_maximization
from expectations import get_market_stats,get_supply
from Utilities.adaptive_expectations import adaptive_expectations, autoregressive
from Utilities.Strategic_adjustments import best_response_exact, get_max_wage,get_min_sale_price, get_desired_capital_price, calculate_production_capacity, get_max_capital_price
import logging
from math import isnan, nan
from Utilities.tools import dict_arithmetic, update_dictionary, calculate_averages
from statistics import mode, StatisticsError

logging.basicConfig(level=logging.INFO)
class Firm(Agent):
    def __init__(self, unique_id, model, initial_capital, initial_productivity):
        super().__init__(unique_id, model)
        self.workers = {}  # Dictionary to store workers and their working hours
        self.total_working_hours = 0
        self.prices = []
        self.strategy = {
          "consumption": {
            "round" : 0,
            "advantage" : ""
          },
          "captial" : {
            "round": 0,
            "advantage": ""},
          "labor":{
            "round": 0,
            "advantage" :""
          }     
        }
        self.capital_prices =[]
        self.sales_same_period =0
        self.debt = 0
        self.zero_profit_conditionsb = {}
        self.a_round_seller = []
        self.market_advantage_seller = []
        self.a_round_buyer = []
        self.market_advantage_buyer = []

        self.production = 0
        self.sales = 0
        self.inventory =0
        self.profit = 0
        self.total_labor_units = 0
        self.market_share = 0
        self.labor_demand = 0
        self.investment_demand = 0
        self.firm_type = ''
        self.mode = 'decentralized'
        self.wage = model.config.INITIAL_WAGE
        self.max_working_hours = model.config.MAX_WORKING_HOURS
        self.zero_profit_conditions = {}
        self.optimals = {}
        self.firm_expectations = {
          'demand': {
            'consumption':[],
            'capital':[],
            'labor':[]
          },
          'price': {
            'consumption':[],
            'capital': [],
            'labor':[]
            },
          'supply':{
            'consumption':[],
            'capital':[],
            'labor':[]

          }
        }
        self.demand_record = {'consumption': [], 'capital': [], 'labor': []}
        self.price_record = {'consumption': [], 'capital': [], 'labor': []}
        self.supply_record = {'consumption': [], 'capital': [], 'labor': []}

        self.demand_expectations = {'consumption': [], 'capital': [], 'labor': []}
        self.price_expectations = {'consumption': [], 'capital': [], 'labor': []}
        self.supply_expectations = {'consumption': [], 'capital': [], 'labor': []}
        self.desireds = {'wage': 0.0625, 'price': 1, 'capital_price': 3}
        self.desireds_record = {'wage': [], 'price': [], 'capital_price': []}
        self.performance_record = {'production': [], 'labor': [], 'capital': [], 'price': [], 'sales': [], 'profit': [], 'debt': [], 'market_share': [], 'budget':[]}
        self.gaps_record = {'production': [], 'labor': [], 'capital': [], 'price': [], 'sales': [], 'profit': [], 'debt': [], 'inventory' : []}
     




    def update_firm_state(self):

        self.depreciation()
        self.performance_review()

        self.sales = 0
        self.production = 0
        self.labor_demand = 0
        self.investment_demand = 0
        self.profit = 0

        self.total_productivity = self.productivity + self.labor_productivity
        self.update_average_price()
        self.prices = []
        self.capital_prices = []

    def depreciation(self):
      #Capital Depreciation
      capital = self.capital * self.model.config.DEPRECIATION_RATE
      self.capital = max(0, self.capital - capital)
      #Inventory Holding Costs
      inventory_cost = self.inventory * self.model.config.HOLDING_COST
      self.budget = max(0, self.budget - inventory_cost)
      return

    def performance_review(self):
      if self.model.step_count <2:
         return
         #Bottle Necks
      performance_actuals ={
        "production" :  self.production,
        "labor" : self.get_total_labor_units(),
        "capital" : self.capital,
        "price" : self.price,
        "sales" : self.sales_same_period,
        "profit" :  self.profit,
        "inventory": self.inventory,
        "debt" :  self.debt,
        "budget": self.budget}

      gaps = dict_arithmetic(self.optimals, performance_actuals, lambda x, y: x - y)
      key = self.firm_type

      match self.firm_expectations['demand'][key][-1:][0], (self.optimals['sales'] - gaps['production']):
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
      
      print(f"rounds seller: {self.a_round_seller}, rounds buyer: {self.a_round_buyer}") 

      #clear lists
      self.a_round_seller = []
      self.market_advantage_seller = [] 
      self.a_round_buyer = []
      self.market_advantage_buyer = []


      return performance_actuals, gaps

    def update_expectations(self):
        # Grab the demand for relavent goods
        consumption_market_stats = get_market_stats(self, 'consumption')
        capital_market_stats = get_market_stats(self, 'capital')
        labor_market_stats = get_market_stats(self, 'labor')

        self.strategy = {
          'consumption': consumption_market_stats,
          'capital': capital_market_stats,
          'labor': labor_market_stats
        }
      

        Demand ={
          'consumption': consumption_market_stats['demand'],
          'capital': capital_market_stats['demand'],
          'labor': labor_market_stats['demand']
        }
        Price = {
          'consumption': consumption_market_stats['price'],
          'capital': capital_market_stats['price'],
          'desired_price': self.desireds['price'],
          'labor': labor_market_stats['price'],
          'desired_wage': self.desireds['wage'],
          'labor_avg_buyer_max_price': labor_market_stats['avg_buyer_max_price'],
          'labor_avg_seller_min_price': labor_market_stats['avg_seller_min_price'],
          'labor_avg_buyer_price': labor_market_stats['avg_buyer_price'],
          'labor_avg_seller_price': labor_market_stats['avg_seller_price'],
          'consumption_avg_buyer_max_price': consumption_market_stats['avg_buyer_max_price'],
          'consumption_avg_buyer_price': consumption_market_stats['avg_buyer_price'],
          'consumption_avg_seller_min_price': consumption_market_stats['avg_seller_min_price'],
          'consumption_avg_seller_price': consumption_market_stats['avg_seller_price'],
          'capital_avg_buyer_max_price': capital_market_stats['avg_buyer_max_price'],
          'capital_avg_buyer_price': capital_market_stats['avg_buyer_price'],
          'capital_avg_seller_min_price': capital_market_stats['avg_seller_min_price'],
          'capital_avg_seller_price': capital_market_stats['avg_seller_price']
        }
        Supply = {
          'consumption': consumption_market_stats['supply'],
          'capital': capital_market_stats['supply'],
          'labor': labor_market_stats['supply']
        }


        # Non Trivial Error: Find out why buyers bid higher than max price.
        
        """if Price['consumption_avg_buyer_max_price'] + 1e-3< Price['consumption_avg_buyer_price'] :
          print("consumption_avg_buyer_max_price < consumption_avg_buyer_price", consumption_market_stats)
          breakpoint()"""

        self.price_record = update_dictionary(Price, self.price_record)
        self.demand_record = update_dictionary(Demand,self.demand_record)
        self.supply_record = update_dictionary(Supply, self.supply_record)

        if self.model.step_count < 2:
            return

        for category in ['demand', 'price', 'supply']:
            historical_data = (
                self.demand_record if category == 'demand' else
                self.price_record if category == 'price' else
                self.supply_record
            )
            new_forecasts = autoregressive(
                historical_data,
                self.firm_expectations[category],
                self.model.time_horizon - 1  # Forecast one less step
            )
            
            # Combine most recent actual data with forecasts
            self.firm_expectations[category] = {
                key: [historical_data[key][-1]] + new_forecasts[key].tolist()
                for key in new_forecasts
            }

        return self.firm_expectations


    def make_production_decision(self):
        match self.firm_type:
          case 'capital':
            return
          case _:
            pass

        average_capital_price = self.model.data_collector.get_average_capital_price(self.model)
        number_of_firms = sum(1 for agent in self.model.schedule.agents if isinstance(agent, (Firm2)))


        Profit_max_params = {
            'current_capital': round(self.capital,2),
            'current_labor': round(self.total_labor_units,2),
            'current_price': round(self.price,2),
            'productivity': self.productivity,
            'expected_demand': list(map(lambda x: x / number_of_firms, self.firm_expectations['demand']['consumption'])),
            'expected_price': self.firm_expectations['price']['consumption'],
            'capital_price': self.firm_expectations['price']['capital'][-1:][0],
            'capital_elasticity': self.capital_elasticity,
            'inventory': round(self.inventory,2),
            'depreciation_rate': self.model.config.DEPRECIATION_RATE,
            'time_horizon': (self.model.time_horizon),
            'discount_rate': self.model.config.DISCOUNT_RATE,
            'budget': round(self.budget,2),
            'wage': self.firm_expectations['price']['labor'][0],
            'expected_capital_supply': self.firm_expectations['supply']['capital'][0],
            'expected_labor_supply': self.firm_expectations['supply']['labor'][0],
            'debt': self.debt,
            'carbon_intensity': self.carbon_intensity,
            'new_capital_carbon_intensity': 1,
            'carbon_tax_rate': 0,
            'holding_costs': self.model.config.HOLDING_COST
        }
        print("Calling profit_maximization with parameters:" , Profit_max_params)

        result, zero_profit_result = profit_maximization(Profit_max_params)

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
        print("optimals", self.optimals)
        if zero_profit_result is not None:
          self.zero_profit_conditionsb = zero_profit_result
          print("actual results ", self.zero_profit_conditionsb)
        else: 
          self.zero_profit_conditionsb = None
        return self.optimals

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

    def hire_worker(self, worker, wage, hours, a_round, market_advantage):
        if hours < 0:
          print("Negative hours")
          breakpoint()
        self.a_round_buyer.append(a_round)
        self.market_advantage_buyer.append(market_advantage)
        if worker in self.workers:
            #print("Worker already hired incresing hours")
            self.update_worker_hours(worker, hours, a_round, market_advantage)
        else:
            #print("Hiring new Worker")
            self.workers[worker] = {'hours':hours, 'wage':wage}
            self.total_working_hours += hours
            worker.get_hired(self, wage, hours, a_round, market_advantage)

    def update_worker_hours(self, worker, hours, a_round, market_advantage):
        if hours < 0:
          print("Negative hours")
          breakpoint()
        if worker in self.workers:
            self.workers[worker]['hours'] += hours
            self.total_working_hours += hours
            worker.update_hours(self, hours, a_round, market_advantage)

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

    def buy_capital(self, quantity, price, a_round, market_advantage):
        if isinstance(self, Firm2):
            self.capital += quantity
            self.investment_demand -= quantity
            budget_change = quantity * price
            #optimal_debt = self.optimals['debt']
           # if optimal_debt - self.debt  <= 0:
              #if not taking debt, purely finance out of budget.
            self.budget -= budget_change
            self.capital_prices.append(price)
            #else:
              #if optimal_debt<= budget_change:
                #self.budget -= budget_change - optimal_debt
                #self.debt += optimal_debt
                #else:
                #self.debt += budget_change
    def sell_goods(self, quantity, price, a_round, market_advantage):
        inventory_change = self.inventory - quantity
        self.a_round_seller.append(a_round) 
        self.market_advantage_seller.append(market_advantage)
        if a_round == 2 and market_advantage == 1:
          breakpoint()
        self.inventory -= quantity
        self.sales += quantity
        budget_change = quantity * price
        self.budget += quantity * price  # Price already adjusted in execute_capital_market
        self.prices.append(price)
        self.sales_same_period += quantity

    def calculate_profit(self):
      revenues = self.sales_same_period* np.mean(self.prices)
      expenses = self.get_total_labor_units() * self.calculate_average_wage() + self.investment_demand* np.mean(self.capital_prices) + self.inventory * self.model.config.HOLDING_COST

      profit = revenues - expenses
      self.profit = profit
      return self.profit

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
          price_expectations = self.firm_expectations['price']['consumption']
          desired_capital_price = get_desired_capital_price(self)
        #for capital firms
        case 'capital', _:
          price_expectations = self.firm_expectations['price']['capital']
          desired_capital_price = 0
        # Doesn't happen but just for pyright to not show red squiggly lines
        case _,_:
          print('invalid firm type')
          breakpoint()
          return

      wage_expectations = self.firm_expectations['price']['labor']

      records = [self.price_record, self.demand_record, self.supply_record]
      for record in records:
        assert record is not None

      price_params = {
        'is_buyer': False,
        'market_type': 'consumption',
        'round_num': self.strategy['consumption']['round_num'],
        'price': self.strategy['consumption']['price'],
        'avg_buyer_price': self.strategy['consumption']['avg_buyer_price'],
        'avg_seller_price': self.strategy['consumption']['avg_seller_price'],
        'avg_seller_min_price': self.strategy['consumption']['avg_seller_min_price'],
        'avg_buyer_max_price': self.strategy['consumption']['avg_buyer_max_price'], 
        'demand': self.strategy['consumption']['demand'],
        'supply': self.strategy['consumption']['supply'],
        'pvt_res_price': self.zero_profit_conditions['price'],
        'previous_price': self.desireds['price']
        }
      desired_price = best_response_exact(price_params, debug = False)

      if desired_price > price_params['avg_buyer_max_price']:
        print("price_params", price_params)
        breakpoint()

      if self.strategy['consumption']['avg_buyer_max_price'] < desired_price:
        print("desired_price", desired_price)
        print("strategy", self.strategy['consumption'])
        print("price params", price_params)
        breakpoint()
      wage_params = { 
        'is_buyer': True,
        'market_type': 'labor',
        'round_num': self.strategy['labor']['round_num'],
        'price': self.strategy['labor']['price'],
        'avg_buyer_price': self.strategy['labor']['avg_buyer_price'],
        'avg_seller_price': self.strategy['labor']['avg_seller_price'],
        'avg_seller_min_price': self.strategy['labor']['avg_seller_min_price'],
        'avg_buyer_max_price': self.strategy['labor']['avg_buyer_max_price'],  
        'demand': self.strategy['labor']['demand'],
        'supply': self.strategy['labor']['supply'],
        'pvt_res_price': self.zero_profit_conditions['wage'],
        'previous_price': self.desireds['wage']
      }
      desired_wage = max(best_response_exact(wage_params, debug = True), self.model.config.MINIMUM_WAGE)

      self.desireds = {
        'wage': desired_wage,
        'price': desired_price,
        'capital_price': desired_capital_price}

      self.desireds_record = update_dictionary(self.desireds, self.desireds_record)


      return self.desireds


    def get_zero_profit_conditions(self):
      """if self.zero_profit_conditionsb is not None:
         self.zero_profit_conditions = {
          'wage': self.zero_profit_conditionsb['wage']/16 + 0.01,
          'price': self.zero_profit_conditionsb['price'] - 0.01,
          'capital_price': self.zero_profit_conditionsb['capital_price']
         }
         if self.zero_profit_conditions['price'] == 0:
            breakpoint()
         print("Stored results ", self.zero_profit_conditions)
         return self.zero_profit_conditions"""
      max_wage = get_max_wage(self.total_working_hours, self.productivity, self.capital, self.capital_elasticity, self.price, self.get_total_labor_units(),self.optimals['labor'],self.model.config.MINIMUM_WAGE)
      min_sale_price = get_min_sale_price(self.firm_type, self.workers, self.productivity, self.capital, self.capital_elasticity, self.get_total_labor_units(), self.inventory)

      if min_sale_price < 0.5:
        breakpoint()

      max_capital_price = get_max_capital_price(self.investment_demand, self.optimals['production'],self.optimals['capital'], self.price, self.capital_elasticity, self.model.time_horizon, self.model.config.DISCOUNT_RATE)
      self.zero_profit_conditions = {
      'wage': max_wage,
      'price': min_sale_price,
      'capital_price': max_capital_price}

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
        self.budget = 10
        self.historic_sales = [model.config.INITIAL_SALES]
        self.historic_inventory = [self.inventory]
        self.expected_demand = [model.config.FIRM1_INITIAL_DEMAND] *self.model.time_horizon
        self.expected_price = [self.price] * self.model.time_horizon
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
        self.historic_sales = [model.config.INITIAL_SALES]
        self.price = self.model.config.INITIAL_PRICE
        self.historic_inventory = [self.inventory]
        self.expected_demand = [model.config.FIRM2_INITIAL_DEMAND]*self.model.time_horizon
        self.expected_price = [self.price]*self.model.time_horizon
        self.firm_expectations ={
          'demand': {
           'consumption': self.expected_demand,
           'capital' : [],
           'labor': []},
          'price':{
          'consumption': self.expected_price,
          'capital':list(map(lambda x: x * self.model.config.INITIAL_RELATIVE_PRICE_CAPITAL, self.expected_price)),
          'labor': [self.model.config.MINIMUM_WAGE] * self.model.time_horizon
          },
          'supply':{
            'consumption': [],
            'capital':[6]*self.model.time_horizon,
            'labor': [300]*self.model.time_horizon
          }}

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
