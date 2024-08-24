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
        self.capital = model.config.INITIAL_CAPITAL
        self.productivity = model.config.INITIAL_PRODUCTIVITY
        self.price = model.config.INITIAL_PRICE
        self.prices = []
        self.inventory = model.config.INITIAL_INVENTORY
        self.historic_demand = [model.config.INITIAL_DEMAND]
        self.historic_price = []
        self.production = 0
        self.sales = 0
        self.firm_type = ''
        self.budget = model.config.INITIAL_CAPITAL
        self.historic_sales = [model.config.INITIAL_SALES]
        self.historic_inventory = [model.config.INITIAL_INVENTORY]
        self.expected_demand = model.config.INITIAL_DEMAND
        self.expected_price = model.config.INITIAL_PRICE
        self.labor_demand = 0
        self.investment_demand = 0
        self.mode = 'decentralized'
        self.wage = model.config.INITIAL_WAGE
        self.max_working_hours = model.config.MAX_WORKING_HOURS

    def step(self):
        self.update_firm_state()
        self.make_production_decision()


    def update_firm_state(self):
       #self.train_demand_predictor()
        depreciation_amount = self.inventory * self.model.config.DEPRECIATION_RATE
        self.pay_wages()
        self.wage = self.calculate_average_wage()
        self.update_historic_sales()
        self.update_historic_prices()
        self.inventory = max(0, self.inventory - depreciation_amount)
        if self.firm_type == 'consumption':
            capital_depreciation = self.capital * self.model.config.DEPRECIATION_RATE
            self.capital = max(0, self.capital - depreciation_amount)
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

    def make_production_decision(self):
        self.production = 0
        buyer_demand, buyer_price = 0, 0
        if isinstance(self, Firm1):
            buyer_demand, buyer_price = self.get_market_demand('capital')
            self.expected_demand =expect_demand(buyer_demand,(self.model.config.TIME_HORIZON - self.model.step_count))
            self.expected_price = expect_price(self.historic_price, self.price, (self.model.config.TIME_HORIZON - self.model.step_count))

        elif isinstance(self, Firm2):
            buyer_demand, buyer_price = self.get_market_demand('consumption')
            self.expected_demand = expect_demand(buyer_demand,(self.model.config.TIME_HORIZON - self.model.step_count))
            self.expected_price = expect_price(self.historic_price, self.price,(self.model.config.TIME_HORIZON - self.model.step_count))

        average_capital_price = self.model.get_average_capital_price()
        print(f"average capital price {average_capital_price}")
        if self.budget < 0:
            print("Budget Exceeded")
            return # Skip production if budget is negative
        print("Calling profit_maximization with parameters:", {
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
        })

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
            (self.model.config.TIME_HORIZON - self.model.step_count),
            self.model.config.DISCOUNT_RATE,
            self.budget,
            self.wage * self.max_working_hours # Per unit wage
        )

        if result is None:
            print("Optimization failed")
            return


        optimal_labor = result['optimal_labor']
        optimal_capital = result['optimal_capital']
        optimal_price = result['optimal_price']
        optimal_production = result['optimal_production']

        print(f"Optimal labor From the Optimisation: {optimal_labor}, optimal capital: {optimal_capital}, optimal price: {optimal_price}, optimal production: {optimal_production}")

        self.labor_demand = max(0, optimal_labor - self.get_total_labor_units()) * self.max_working_hours  # Convert to hours

        if optimal_labor < self.get_total_labor_units():
            self.layoff_employees(self.get_total_labor_units() - optimal_labor)
        print(f"prices pre-optimisation: {self.price}")
        self.price = self.adjust_price()
        print(f"prices post-optimisation: {self.price}")
        self.investment_demand = max(0, optimal_capital - self.capital)
        self.production =  min(optimal_production, self.calculate_production_capacity())

        self.inventory += self.production

    def adjust_price(self) -> float:
        """
        Adjust price based on sales vs expected demand.

        :param current_price: The current price of the good
        :param sales: Actual sales in the last period
        :param expected_demand: Expected demand for the last period
        :return: Adjusted price
        """


        if (self.sales - self.inventory) >= 0: # 1% buffer
            print("Sales Exceeded Production", self.sales - self.production)
            price_hike = np.random.uniform(1.01, 1.1)
            print("Price Hike", price_hike)
            proposed_price =  self.price * price_hike
            return proposed_price # Increase price by 5%
        else:
            print("Sales Shortfall", self.sales - self.production)
            price_cut = np.random.uniform(0.95, 0.99)

            print("Price Cut", price_cut)
            proposed_price = self.price * price_cut
            proposed_price = max(proposed_price, self.get_min_sale_price())
            return proposed_price  # Decrease price by 5%

    def adjust_capital(self, current_capital: float,
                    optimal_capital: float,
                    budget: float,
                    capital_price: float) -> float:
        """
        Adjust capital based on optimal capital and budget constraints.

        :param current_capital: Current capital stock
        :param optimal_capital: Optimal capital stock from profit maximization
        :param budget: Available budget for capital adjustment
        :param capital_price: Price of capital (in labor units)
        :return: New capital stock
        """
        desired_change = optimal_capital - current_capital
        max_affordable_change = budget / max(capital_price,1)

        if desired_change > 0:
            # Buying capital
            actual_change = min(desired_change, max_affordable_change)
        else:
            # Selling capital (assume we can sell any amount)
            return 0

        return actual_change


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
            old_hours = self.workers[worker]['hours']
            self.workers[worker]['hours'] += hours
            self.total_working_hours += hours
            worker.update_hours(self, hours)

    def pay_wages(self):
        fire_list = []

        for worker in self.workers:
            wage = self.workers[worker]['wage']
            hours = self.workers[worker]['hours']
            budget_change = wage * hours
            print(f"Budget Change Labor Cost for firm {self.unique_id}", budget_change)
            if self.budget >= budget_change:
                self.budget -= budget_change
                worker.got_paid = True
            else:
                worker.got_paid = False
                fire_list.append(worker)
                print("Not enough budget to pay wages")
        for worker in fire_list:
            self.fire_worker(worker)

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
            print("Average Wage empty start", wage_avg)
            return wage_avg
        wage_avg = np.mean([self.workers[worker]['wage'] for worker in self.workers])
        print("Average Wage", wage_avg)
        return wage_avg

    def get_total_labor_units(self):
        return self.total_working_hours / self.max_working_hours
    def buy_capital(self, quantity, price):
        if isinstance(self, Firm2):
            self.capital += quantity
            self.investment_demand -= quantity
            print("Budget pre-purchase",self.budget)
            budget_change = quantity * price
            print(f"Budget Change Capital Cost for firm{self.unique_id}", -budget_change)
            self.budget -= budget_change
            print("Capital Bought", quantity)
    def sell_goods(self, quantity, price):
        self.inventory -= quantity
        self.sales += quantity
        self.budget += quantity * price  # Price already adjusted in execute_capital_market
        self.prices.append(price)


    def update_historic_sales(self):
        self.historic_sales.append(self.sales)
        inventory_change = self.production - self.sales
        self.historic_inventory.append(inventory_change)
        self.sales = 0


    def update_historic_prices(self):
        mean_price = self.price
        if len(self.prices) > 0:
           mean_price = np.mean(self.prices)
        self.historic_price.append(mean_price)

    def get_max_wage(self):
        if self.labor_demand <= 0:
            return 0

        # Convert labor_demand to hours
        labor_demand = self.labor_demand

        # Calculate total revenue at optimal production
        total_revenue = self.production * self.price

        # Calculate average revenue product of labor (per hour)
        avg_revenue_product = total_revenue / labor_demand if labor_demand> 0 else 0

        # Set max wage as a fraction of average revenue product
        max_wage_factor = 1  # Allows wages up to x% above average revenue product
        max_wage = avg_revenue_product * max_wage_factor

        # Ensure max wage doesn't exceed budget constraint
        budget_constraint = self.budget / labor_demand if labor_demand > 0 else 0
        max_wage = min(max_wage, budget_constraint)
        max_wage = max(max_wage, self.model.config.MINIMUM_WAGE)  # Allows wages up to x% above average revenue product
        return min(max_wage, budget_constraint)
    def get_min_sale_price(self):

        if self.firm_type == 'consumption':
            labor_cost = sum([self.workers[worker]['wage'] * self.workers[worker]['hours'] for worker in self.workers])
            capital_cost = self.investment_demand * min(self.get_max_capital_price(), self.model.get_average_capital_price())
            total_cost = labor_cost + capital_cost
            total_output = self.inventory
            if total_output <= 0:
                return 0
            return max(total_cost / total_output, 0.5) # harcoding a lower bound for now
        else:
            labor_cost = sum([self.workers[worker]['wage'] * self.workers[worker]['hours'] for worker in self.workers])

            total_output = self.inventory
            if total_output <= 0:
                return 1.5
            if labor_cost <= 0.001:
                return 1.5
            return max(total_output/labor_cost, 1.5) # harcoding a lower bound for now





    def get_max_capital_price(self):
        if self.investment_demand <= 0:
            return 0

        # Optimal values from profit maximization
        optimal_capital = self.capital + self.investment_demand
        optimal_production = self.production
        optimal_price = self.price
        total_revenue = 0
        # Calculate total revenue at optimal production
        for i in range(self.model.config.TIME_HORIZON):
            total_revenue += optimal_production * optimal_price * (1-self.model.config.DISCOUNT_RATE)**i



        # Calculate marginal revenue product of capital
        marginal_revenue_product = (total_revenue / optimal_capital) * self.capital_elasticity

        # Set max capital price as a fraction of marginal revenue product
        max_price_factor = 1.0  # Allows prices up to x% above marginal revenue product
        max_capital_price = marginal_revenue_product * max_price_factor
        print("Max Capital Price", max_capital_price)
        # Ensure max capital price doesn't exceed budget constraint
        #

        budget_constraint = self.budget
        max_capital_price = min(max_capital_price, budget_constraint)
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
    def step(self):
        super().step()
  # Reset price to initial value
        if self.mode == 'decentralized':
            self.innovate()

    def innovate(self):
        if self.model.random.random() < self.model.config.INNOVATION_ATTEMPT_PROBABILITY:
            rd_investment = self.capital * self.model.config.FIRM1_RD_INVESTMENT_RATE
            print(f"Budget Change RD Investment for firm {self.unique_id}", -rd_investment)
            self.budget -= rd_investment
            if self.model.random.random() < self.model.config.PRODUCTIVITY_INCREASE_PROBABILITY:
                self.productivity *= (1 + self.model.config.PRODUCTIVITY_INCREASE)

class Firm2(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM2_INITIAL_CAPITAL, model.config.INITIAL_PRODUCTIVITY)
        self.capital_inventory = 0  # Separate inventory for capital goods
        self.firm_type = 'consumption'
        self.capital_elasticity = model.config.CAPITAL_ELASTICITY_FIRM2
        self.investment_demand = model.config.FIRM2_INITIAL_INVESTMENT_DEMAND
