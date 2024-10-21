import numpy as np
from Turbo_utils import *
class Firm:
    def __init__(self, id, initial_wealth):
        self.id = id
        self.wealth = 5
        self.capital = 6
        self.productivity = 1
        self.capital_elasticity = 0.5
        self.depreciation_rate = 0.1
        self.discount_rate = 0.05
        self.production = 0
        self.labor_demand = 0
        self.labor_bid = 0.0625 
        self.consumption_ask = 1
        self.time_horizon = 1
        self.labor_price_expectations = [1]
        self.labor_supply_expectations = [300]
        self.consumption_price_expectations = [1]
        self.consumption_demand_expectations = [30]

    def get_min_consumption_price(self):
        # Set consumption prices based on labor price expectations
        costs = self.labor_demand * self.labor_price_expectations[0]
        price = costs / self.production
        return price

    def get_max_labor_price(self):
        # Set labor prices based on consumption price expectations
        revenue = self.production * self.consumption_price_expectations[0]
        price = revenue / self.labor_demand
        return price
    
    def get_profit_params(self):
        params = {
            'current_capital': self.capital,
            'current_labor': 0,
            'current_price': self.consumption_price_expectations[0],
            'current_productivity': self.productivity,
            'expected_demand': self.consumption_demand_expectations,
            'expected_price': self.consumption_price_expectations,
            'capital_price': 1,
            'capital_elasticity': self.capital_elasticity,
            'current_inventory': 0,
            'depreciation_rate': self.depreciation_rate,
            'expected_periods': self.time_horizon,
            'discount_rate': self.discount_rate,
            'budget': self.wealth,
            'wage': self.labor_price_expectations,
            'capital_supply': 0,
            'labor_supply': self.labor_supply_expectations
        }
        if not check_valid_params(params):
            raise ValueError("Invalid parameters for profit maximization")
        return params

    def update_expectations(self, labor_market_stats, consumption_market_stats):
        """
        Update firm's expectations based on market statistics.
        
        :param labor_market_stats: Dictionary containing labor market statistics
        :param consumption_market_stats: Dictionary containing consumption market statistics
        """
        # Update labor market expectations
        if is_valid_number(labor_market_stats.get('price')):
            self.labor_price_expectations = [labor_market_stats['price']]
            self.labor_supply_expectations = [labor_market_stats['supply']]
        # Update consumption market expectations
        if is_valid_number(consumption_market_stats.get('price')):
            self.consumption_price_expectations = [consumption_market_stats['price']]
        if is_valid_number(consumption_market_stats.get('demand')):
            self.consumption_demand_expectations = [consumption_market_stats['demand']]
