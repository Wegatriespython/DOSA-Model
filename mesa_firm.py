from mesa import Agent
import numpy as np
from Simple_profit_maxxing import neoclassical_profit_maximization

class Firm(Agent):
    def __init__(self, unique_id, model, initial_capital, initial_productivity):
        super().__init__(unique_id, model)
        self.capital = initial_capital
        self.productivity = initial_productivity
        self.price = model.config.INITIAL_PRICE
        self.inventory = model.config.INITIAL_INVENTORY
        self.workers = []
        self.production = 0
        self.sales = 0
        self.budget = initial_capital
        self.historic_sales = [model.config.INITIAL_DEMAND] * 5  # Initialize with 5 periods of initial demand
        self.expected_demand = model.config.INITIAL_DEMAND
        self.expected_price = model.config.INITIAL_PRICE
        self.labor_demand = 0
        self.investment_demand = 0
        self.mode = 'decentralized'

    def step(self):
        if self.mode == 'decentralized':
            self.decentralized_step()
        elif self.mode == 'centralized':
            self.centralized_step()

    def decentralized_step(self):
        self.update_firm_state()
        self.make_production_decision()
        self.produce()
        self.update_historic_sales()
    def centralized_step(self):
        pass
    def update_firm_state(self):
        if isinstance(self, Firm1):
            self.expected_demand = self.calculate_expected_demand('capital')
        elif isinstance(self, Firm2):
            self.expected_demand = self.calculate_expected_demand('consumption')
        self.expected_price = self.calculate_expected_price()

        depreciation_amount = self.inventory * self.model.config.DEPRECIATION_RATE
        self.inventory = max(0, self.inventory - depreciation_amount)
        self.budget -= depreciation_amount * self.price

    def calculate_expected_demand(self, market_type):
        """
        Calculate expected demand based on historical sales and average buyer demand.

        :param market_type: String indicating the market type ('labor', 'capital', or 'consumption')
        :return: Expected demand
        """
        historical_demand = np.mean(self.historic_sales[-5:])

        if market_type == 'labor':
            potential_buyers = [agent for agent in self.model.schedule.agents if isinstance(agent, Firm)]
            buyer_demand = [firm.labor_demand for firm in potential_buyers]
        elif market_type == 'capital':
            potential_buyers = [agent for agent in self.model.schedule.agents if isinstance(agent, Firm)]
            buyer_demand = [firm.investment_demand for firm in potential_buyers]
        elif market_type == 'consumption':
            potential_buyers = [agent for agent in self.model.schedule.agents if hasattr(agent, 'consumption')]
            buyer_demand = [agent.consumption for agent in potential_buyers]
        else:
            raise ValueError(f"Invalid market type: {market_type}")

        average_buyer_demand = np.mean(buyer_demand) if buyer_demand else 0

        # Combine historical demand and average buyer demand with 50% bias towards historical
        combined_demand = 0.5 * historical_demand + 0.5 * average_buyer_demand

        # Ensure the expected demand is not below a minimum threshold
        return max(combined_demand, self.model.config.MIN_DEMAND)


    def make_production_decision(self):
        # Update expected demand first
        if isinstance(self, Firm1):
            self.expected_demand = self.calculate_expected_demand('capital')
        elif isinstance(self, Firm2):
            self.expected_demand = self.calculate_expected_demand('consumption')

        optimal_labor, optimal_capital, optimal_price, optimal_production = neoclassical_profit_maximization(
            self.capital,
            len(self.workers),
            self.price,
            self.productivity,
            [self.expected_demand] * 5,
            [self.expected_price] * 5,
            self.model.global_accounting.get_average_capital_price(),
            self.model.config.CAPITAL_ELASTICITY,
            self.inventory,
            self.model.config.DEPRECIATION_RATE,
            5,
            self.model.config.DISCOUNT_RATE
        )
        print(f"Optimal labor: {optimal_labor}, optimal capital: {optimal_capital}, optimal price: {optimal_price}, optimal production: {optimal_production}")

        self.labor_demand = max(0, optimal_labor - len(self.workers))
        self.price = self.adjust_price(self.price, self.sales, self.expected_demand)
        self.investment_demand = self.adjust_capital(self.capital, optimal_capital, self.budget, self.model.global_accounting.get_average_capital_price())
        self.production = self.adjust_production(self.production, optimal_production)

    def adjust_production(self, current_production: float, optimal_production: float, max_adjustment_rate: float = 0.1) -> float:
        """
        Adjust production gradually towards the optimal level.

        :param current_production: Current production level
        :param optimal_production: Optimal production level from profit maximization
        :param max_adjustment_rate: Maximum rate of change in production (default 10%)
        :return: New production level
        """
        production_difference = optimal_production - current_production
        max_change = 5

        if abs(production_difference) <= max_change:
            # If the difference is small, move directly to optimal
            print("Optimal production", optimal_production)
            return optimal_production
        elif production_difference > 0:
            # If we need to increase production
            print("Current production", current_production + max_change)
            return current_production + max_change
        else:
            # If we need to decrease production
            return current_production - max_change

    def adjust_price(self, current_price: float, sales: float, expected_demand: float) -> float:
        """
        Adjust price based on sales vs expected demand.

        :param current_price: The current price of the good
        :param sales: Actual sales in the last period
        :param expected_demand: Expected demand for the last period
        :return: Adjusted price
        """
        if sales >= expected_demand:
            proposed_price = current_price * 1.05
            return proposed_price # Increase price by 5%
        else:
            proposed_price = current_price * 0.95
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

    def produce(self):
        if isinstance(self, Firm1):
            self.produce_capital_goods()
        elif isinstance(self, Firm2):
            self.produce_consumption_goods()

    def produce_capital_goods(self):
        # Implement production function for Firm1
        self.production = self.productivity * (self.capital ** self.model.config.CAPITAL_ELASTICITY) * (len(self.workers) ** (1 - self.model.config.CAPITAL_ELASTICITY))
        self.inventory += self.production

    def produce_consumption_goods(self):
        # Implement production function for Firm2
        capital_input = min(self.capital, self.inventory)  # Can't use more capital than available
        self.production = self.productivity * (capital_input ** self.model.config.CAPITAL_ELASTICITY) * (len(self.workers) ** (1 - self.model.config.CAPITAL_ELASTICITY))
        self.inventory += self.production
        self.capital -= capital_input  # Consume the capital goods used in production

    def update_historic_sales(self):
        self.historic_sales.append(self.sales)
        if len(self.historic_sales) > 10:
            self.historic_sales.pop(0)
        self.sales = 0

    def calculate_expected_price(self):
        return np.mean([self.price] + [self.model.global_accounting.get_average_consumption_good_price() for _ in range(4)])

    def calculate_production_capacity(self):
        return self.productivity * (self.capital ** self.model.config.CAPITAL_ELASTICITY) * (len(self.workers) ** (1 - self.model.config.CAPITAL_ELASTICITY))

    def hire_worker(self, worker, wage):
        self.workers.append(worker)
        self.budget -= wage
        self.labor_demand -= 1

    def fire_workers(self, num_workers):
        for _ in range(min(num_workers, len(self.workers))):
            worker = self.workers.pop()
            worker.get_fired()

    def buy_capital(self, quantity, price):
        if isinstance(self, Firm2):
            self.capital += quantity
            self.investment_demand -= quantity
            self.budget -= quantity * price * self.model.relative_price  # Adjust for relative price

    def sell_goods(self, quantity, price):
        self.inventory -= quantity
        self.sales += quantity
        if isinstance(self, Firm1):
            self.budget += quantity * price * self.model.relative_price  # Adjust for relative price
        else:
            self.budget += quantity * price  # Consumption goods are the numeraire

    def get_max_wage(self):
        return self.budget / max(1, self.labor_demand) if self.labor_demand > 0 else 0

    def get_max_capital_price(self):
        return self.budget / max(1, self.investment_demand) if self.investment_demand > 0 else 0

    def apply_central_decision(self, labor, capital, price):
        self.labor_demand = labor
        self.capital = capital
        if isinstance(self, Firm1):
            self.price = price * self.model.relative_price  # Adjust for relative price
        else:
            self.price = price  # Consumption goods are the numeraire
        self.produce()
    def update_after_markets(self):
        pass

class Firm1(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM1_INITIAL_CAPITAL, model.config.INITIAL_PRODUCTIVITY)

    def step(self):
        super().step()
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
    def buy_capital(self, quantity, price): # SUS
        self.capital_inventory += quantity
        self.investment_demand -= quantity
        self.budget -= quantity * price * self.model.relative_price  # Adjust for relative price
    def produce_consumption_goods(self):
        capital_input = min(self.capital_inventory, self.model.config.MAX_CAPITAL_USAGE)
        self.production = self.productivity * (capital_input ** self.model.config.CAPITAL_ELASTICITY) * (len(self.workers) ** (1 - self.model.config.CAPITAL_ELASTICITY))
        self.inventory += self.production
        self.capital_inventory -= capital_input  # Consume the capital goods used in production
