from mesa import Agent
import numpy as np
import random

class Firm(Agent):
    def __init__(self, unique_id, model, initial_capital, initial_rd_investment):
        super().__init__(unique_id, model)
        self.capital = initial_capital
        self.productivity = model.config.INITIAL_PRODUCTIVITY
        self.price = model.config.INITIAL_PRICE
        self.inventory = model.config.INITIAL_INVENTORY
        self.workers = []
        self.demand = model.config.INITIAL_DEMAND
        self.production = 0
        self.sales = 0
        self.budget = 0
        self.RD_investment = initial_rd_investment
        self.expected_demand = model.config.INITIAL_DEMAND
        self.labor_demand = 0
        self.investment_demand = 0

    def step(self):
        self.optimize_production()
        self.produce()

    def optimize_production(self):
        optimal_labor, optimal_capital, optimal_price, optimal_production = self.simple_profit_maximization()
        self.adjust_labor(optimal_labor)
        self.adjust_capital(optimal_capital)
        self.price = optimal_price
        self.production = optimal_production

    def simple_profit_maximization(self):
        max_profit = float('-inf')
        optimal_labor = len(self.workers)
        optimal_capital = self.capital
        optimal_price = self.price 
        optimal_production = 0

        expected_demand = self.calculate_expected_demand()
        current_wage_bill = sum(worker.wage for worker in self.workers)

        available_budget = max(0, self.budget - current_wage_bill)
        avg_wage = self.model.get_average_wage()
        avg_capital_price = self.model.get_average_capital_price()
        
        # Define realistic ranges for optimization
        labor_range = range(max(0, len(self.workers) - 2), min(len(self.workers) + 3, int(available_budget / avg_wage) + 1))
        capital_range = range(
            max(1, int(self.capital * 0.8)),  # Allow for some disinvestment
            min(int(self.capital * 1.2), int(self.capital + available_budget // avg_capital_price) + 1)
        )
        price_range = np.linspace(self.price * 0.8, self.price * 1.2, 20)  # 20 price points between 80% and 120% of current price

        for L in labor_range:
            for K in capital_range:
                for P in price_range:
                    Q = min(self.cobb_douglas_production(L, K), expected_demand)
                    revenue = P * Q
                    
                    labor_cost = L * avg_wage
                    capital_cost = max(0, K - self.capital) * avg_capital_price
                    total_cost = labor_cost + capital_cost

                    if total_cost > available_budget:
                        continue  # Skip if the combination is not affordable

                    profit = revenue - total_cost

                    if profit > max_profit:
                        max_profit = profit
                        optimal_labor = L
                        optimal_capital = K
                        optimal_price = P
                        optimal_production = Q

        # Ensure we don't return negative values
        return max(0, optimal_labor), max(1, optimal_capital), max(0.01, optimal_price), max(0, optimal_production)

    def adjust_labor(self, optimal_labor):
        current_labor = len(self.workers)
        if optimal_labor > current_labor:
            self.labor_demand = optimal_labor - current_labor
        elif optimal_labor < current_labor:
            workers_to_fire = current_labor - optimal_labor
            for _ in range(int(workers_to_fire)):
                if self.workers:
                    worker = self.workers.pop()
                    worker.employed = False
                    worker.employer = None
                    worker.wage = 0
        else:
            self.labor_demand = 0

    def adjust_capital(self, optimal_capital):
        capital_difference = optimal_capital - self.capital
        if capital_difference > 0:
            capital_to_buy = min(capital_difference, self.budget // self.model.get_average_capital_price())
            self.investment_demand = capital_to_buy
        elif capital_difference < 0:
            self.investment_demand = 0

    def produce(self):
        self.inventory += self.production

    def calculate_expected_demand(self):
        return self.model.config.DEMAND_ADJUSTMENT_RATE * self.model.average_market_demand + (1 - self.model.config.DEMAND_ADJUSTMENT_RATE) * self.demand

    def cobb_douglas_production(self, labor, capital):
        return self.productivity * (capital ** self.model.config.CAPITAL_ELASTICITY) * (labor ** (1 - self.model.config.CAPITAL_ELASTICITY))

class Firm1(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM1_INITIAL_CAPITAL, model.config.FIRM1_INITIAL_RD_INVESTMENT)

    def step(self):
        self.innovate()
        super().step()

    def innovate(self):
        if random.random() < self.model.config.INNOVATION_ATTEMPT_PROBABILITY:
            self.RD_investment = self.capital * self.model.config.FIRM1_RD_INVESTMENT_RATE
            self.budget -= self.RD_investment
            if random.random() < self.model.config.PRODUCTIVITY_INCREASE_PROBABILITY:
                self.productivity *= (1 + self.model.config.PRODUCTIVITY_INCREASE)

class Firm2(Firm):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, model.config.FIRM2_INITIAL_CAPITAL, model.config.FIRM2_INITIAL_INVESTMENT_DEMAND)
        self.investment_demand = model.config.FIRM2_INITIAL_INVESTMENT_DEMAND
        self.investment = 0
        self.desired_capital = model.config.FIRM2_INITIAL_DESIRED_CAPITAL