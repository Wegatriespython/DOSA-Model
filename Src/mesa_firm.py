from mesa import Agent
import numpy as np
from Utilities.Simple_profit_maxxing import profit_maximization
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from Utilities.expectations import expect_demand, expect_price
import logging

logging.basicConfig(level=logging.INFO)
class Firm(Agent):
    def __init__(self, unique_id, model, initial_capital, initial_productivity):
        super().__init__(unique_id, model)
        self.workers = {}  # Dictionary to store workers and their working hours
        self.total_working_hours = 0
        self.prices = []
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
        self.expectations = 0
        self.pay_wages()




        print(f"firm id {self.unique_id} budget {self.budget}")

    def get_market_demand(self, market_type):
        if market_type == 'labor':
            potential_buyers = [agent for agent in self.model.schedule.agents if isinstance(agent, Firm)]
            buyer_demand = [firm.labor_demand for firm in potential_buyers]
            buyer_demand = sum(buyer_demand)
            buyer_price = [firm.wage for firm in potential_buyers]
            buyer_price = np.mean(buyer_price)
        elif market_type == 'capital':
            potential_buyers = [agent for agent in self.model.schedule.agents if isinstance(agent, Firm2)]
            buyer_demand =[firm.investment_demand for firm in potential_buyers]
            buyer_demand = sum(buyer_demand)/2
            buyer_price = [firm.get_max_capital_price() for firm in potential_buyers]
            buyer_price = np.mean(buyer_price)
        elif market_type == 'consumption':
            potential_buyers = [agent for agent in self.model.schedule.agents if hasattr(agent,'consumption')]
            buyer_demand = [agent.desired_consumption for agent in potential_buyers]
            buyer_demand = sum(buyer_demand)/5
            buyer_price = [agent.expected_price for agent in potential_buyers]
            buyer_price = np.mean(buyer_price)

        else:
            raise ValueError(f"Invalid market type: {market_type}")
        avg_price = buyer_price
        self.historic_demand.append(buyer_demand)

        return buyer_demand, buyer_price


    def get_expected_demand(self):

        buyer_demand, buyer_price = 0, 0
        if isinstance(self, Firm1):
            buyer_demand, buyer_price = self.get_market_demand('capital')
            self.expected_demand =expect_demand(buyer_demand,(self.model.config.TIME_HORIZON))
            self.expected_price = expect_price(buyer_price, (self.model.config.TIME_HORIZON))
            self.expectations=[np.mean(self.expected_demand), np.mean(self.expected_price)]
            self.expectations_cache.append(self.expectations)

            if len(self.expectations_cache)>5:
                self.expectations_cache = self.expectations_cache[-5:]

            self.expectations = np.mean(self.expectations, axis=0)
        elif isinstance(self, Firm2):
            buyer_demand, buyer_price = self.get_market_demand('consumption')
            self.expected_demand = expect_demand(buyer_demand,(self.model.config.TIME_HORIZON))
            self.expected_price = expect_price(buyer_price, (self.model.config.TIME_HORIZON))
            self.expectations=[np.mean(self.expected_demand), np.mean(self.expected_price)]

        return self.expected_demand, self.expected_price

    def make_production_decision(self):
        self.historic_sales.append(self.sales)
        if len(self.historic_sales)>5:
            self.historic_sales = self.historic_sales[-5:]


        average_capital_price = self.model.get_average_capital_price()

        if self.budget < 0:

            return # Skip production if budget is negative
        """print("Calling profit_maximization with parameters:", {
            'current_capital': self.capital,
            'current_labor': self.get_total_labor_units(),
            'current_price': self.price,
            'current_productivity': self.productivity,
            'expected_demand': self.expected_demand,
            'expected_price': self.expected_price,
            'capital_price': self.model.get_average_capital_price(),  # Updated
            'capital_elasticity': self.capital_elasticity,
            'current_inventory': self.inventory,
            'depreciation_rate': self.model.config.DEPRECIATION_RATE,
            'expected_periods': (self.model.config.TIME_HORIZON - self.model.step_count),
            'discount_rate': self.model.config.DISCOUNT_RATE,
            'budget': self.budget,
            'wage': self.wage * self.max_working_hours # Per unit wage
        })"""

        result = profit_maximization(
            self.capital,
            self.get_total_labor_units(),
            self.price,
            self.productivity,
            self.expected_demand,
            self.expected_price,
            self.model.get_average_capital_price(),  # Updated
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
                self.capital_inventory = self.capital - optimal_capital
                self.capital_resale_price = self.model.get_average_capital_price()
                self.captial_min_price = 0.1
            return self.investment_demand
    def adjust_production(self):
        if self.inventory > self.model.config.INVENTORY_THRESHOLD:
            self.production = 0
            return self.production
        optimal_production = self.optimals[2]
        self.production =  min(optimal_production, self.calculate_production_capacity())

        self.inventory += max(0, self.production)
        return self.production
    def adjust_price(self) -> float:
        import logging
        logging.info(f"Adjusting price for {self.unique_id}. Current price: {self.price}, Inventory: {self.inventory}, Optimal inventory: {self.optimals[3]}")
        if self.optimals[2] < 0.5:
            return self.expectations[1]
        if self.production < 1 and self.inventory < self.model.config.INVENTORY_THRESHOLD:
            return self.expectations[1]
        recent_sales = np.mean(self.historic_sales)

        if round(recent_sales) < round(self.optimals[4]):# 5% Shortfall in sales leads to price cuts
            price_cut = np.random.uniform(0.95, 0.99)
            proposed_price = self.expectations[1] * round(price_cut, 2)
            proposed_price = max(proposed_price, self.get_min_sale_price())
            self.price = proposed_price
            logging.info(f"Price decreased. New price: {self.price}")
        elif round(self.optimals[2]) > 1:
            price_hike = np.random.uniform(1.01, 1.1)
            logging.info(f"Price Hike for {self.unique_id}: {price_hike}")
            proposed_price = self.expectations[1]* round(price_hike, 2)
            self.price = proposed_price
            logging.info(f"Price increased. New price: {self.price}")

        logging.info(f"Final adjusted price for {self.unique_id}: {self.price}")
        return self.price



    def calculate_production_capacity(self):
        return self.productivity * (self.capital ** self.capital_elasticity) * (self.total_working_hours ** (1 - self.capital_elasticity))

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
    def wage_hikes(self):
        if self.model.step_count == 1000:
            for worker in self.workers:
                if worker.skills > np.mean([worker.skills for worker in self.workers]):
                    wage = worker.expected_wage
                    self.workers[worker]['wage'] = wage




    def pay_wages(self):
        fire_list = []
        wage_total = 0
        employees_total=0
        skill_total = 0
        self.wage_hikes()
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

    def fire_worker(self, worker):
        if worker in self.workers:
            hours = self.workers[worker]['hours']
            self.total_working_hours -= hours
            del self.workers[worker]
            worker.get_fired(self)


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

    def calculate_average_wage(self):
        if self.workers == {}:
            wage_avg = self.model.get_average_wage()
            wage_avg = max(wage_avg, self.model.config.MINIMUM_WAGE)
            return wage_avg
        wage_avg = np.mean([self.workers[worker]['wage'] for worker in self.workers])
        wage_avg = max(wage_avg, self.model.config.MINIMUM_WAGE)
        return wage_avg
    def get_desired_capital_price(self):
        average_capital_price = self.model.get_average_capital_price()
        return average_capital_price

    def get_total_labor_units(self):
        return self.total_working_hours / self.max_working_hours
    def buy_capital(self, quantity, price):
        if isinstance(self, Firm2):
            self.capital += quantity
            self.investment_demand -= quantity

            budget_change = quantity * price

            self.budget -= budget_change

    def sell_goods(self, quantity, price):
        inventory_change = self.inventory - quantity
        print(f"Inventory change: {inventory_change}, sales {quantity}")
        self.inventory -= quantity
        self.sales += quantity
        self.budget += quantity * price  # Price already adjusted in execute_capital_market
        self.prices.append(price)

    def update_historic_prices(self):
        mean_price = self.price
        if len(self.prices) > 0:
           mean_price = np.mean(self.prices)
        self.historic_price.append(mean_price)

    def get_max_wage(self):
        total_output = self.calculate_production_capacity() + self.inventory

        # Calculate marginal product of labor
        labor_units = self.get_total_labor_units()
        if labor_units > 0:
            marginal_product = total_output / labor_units
        else:
            return self.model.config.MINIMUM_WAGE

        max_wage = marginal_product
                # Minimum wage is set by model config
        min_wage = self.model.config.MINIMUM_WAGE
                # Ensure max wage is not lower than min wage
        max_wage = max(max_wage, min_wage)

        if self.optimals[0] > self.get_total_labor_units():
            wage_hike = min_wage + (max_wage - self.wage) * 0.1

        wage_hike = max(min_wage, wage_hike)
        return wage_hike

    def get_min_sale_price(self):

        if self.firm_type == 'consumption':
            labor_cost = sum([self.workers[worker]['wage'] * self.workers[worker]['hours'] for worker in self.workers])
            capital_cost = self.capital * min(self.get_max_capital_price(), self.model.get_average_capital_price())
            total_cost = labor_cost + capital_cost
            total_output = self.calculate_production_capacity()+ self.inventory
            if total_output <= 0:
                return 0.5
            if total_cost <= 0.001:
                return 0.5
            return total_cost / total_output # harcoding a lower bound for now
        else:
            labor_cost = self.total_working_hours * self.calculate_average_wage()

            total_output = self.calculate_production_capacity() + self.inventory
            if total_output <= 0:
                return 1.5
            if labor_cost <= 0.001:
                return 1.5
            return labor_cost/total_output # harcoding a lower bound for now


    def get_max_capital_price(self):
        if self.investment_demand <= 0:
            return 0

        # Optimal values from profit maximization
        optimal_production = self.optimals[2]
        optimal_price = self.price
        optimal_capital = self.optimals[1]
        total_revenue = 0
        # Calculate total revenue at optimal production
        for i in range(self.model.config.TIME_HORIZON):
            total_revenue += optimal_production * optimal_price * (1-self.model.config.DISCOUNT_RATE)**i



        # Calculate marginal revenue product of capital
        marginal_revenue_product = (total_revenue / optimal_capital) * self.capital_elasticity

        # Set max capital price as a fraction of marginal revenue product
        max_price_factor = 1.2  # Allows prices up to x% above marginal revenue product
        max_capital_price = marginal_revenue_product * max_price_factor


        return max_capital_price


    def update_after_markets(self):
        pass
    def get_market_type(self):
        if isinstance(self, Firm1):
            return 'capital'
        elif isinstance(self, Firm2):
            return 'consumption'
        else:
            return 'labor'

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
        self.budget = self.capital
        self.historic_sales = [model.config.INITIAL_SALES]
        self.price = self.model.config.INITIAL_PRICE
        self.historic_inventory = [self.inventory]
        self.expected_demand = model.config.FIRM2_INITIAL_DEMAND
        self.expected_price = self.price
