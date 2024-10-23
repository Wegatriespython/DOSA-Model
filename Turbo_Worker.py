import numpy as np
from Turbo_utils import *
class Worker:
    def __init__(self, unique_id, initial_wealth):
        self.unique_id = unique_id
        self.savings = initial_wealth
        self.max_working_hours = 16
        self.working_hours = 0
        self.total_working_hours = 0
        self.desired_consumption = 1
        self.desired_wage = 1
        self.desired_price = 1
        self.consumption = 0
        self.wage = 1
        self.time_horizon = 1
        self.worker_expectations = {
            'demand': {
                'labor': [300],
                'consumption': [30]
            },
            'price': {
                'labor': [.0625],
                'consumption': [1]
            },
            'supply': {
                'labor': [300],
                'consumption': [30]
            },
            'profit': {
                'consumption': [0]
            }
        }
        self.demand_record = {'labor': [], 'consumption': []}
        self.price_record = {'labor': [], 'consumption': []}
        self.supply_record = {'labor': [], 'consumption': []}

    def get_max_consumption_price(self):
        # set prices based on wage expectations
        price = self.worker_expectations['price']['consumption'][0] * 1.5
        return price

    def get_min_wage(self):
        # Set wages based on consumption price expectations
        wage = self.worker_expectations['price']['labor'][0] * .5
        wage = max(wage, 0.04)
        return wage


    def get_utility_params(self):
        params = {
            'savings': self.savings,
            'wage': self.worker_expectations['price']['labor'],
            'price': self.worker_expectations['price']['consumption'],
            'discount_rate': 0.05,
            'time_horizon': self.time_horizon,
            'alpha': 0.9,
            'profit_income': self.worker_expectations['profit']['consumption'],
            'max_working_hours': self.max_working_hours,
            'working_hours': self.working_hours,
            'expected_labor_demand': self.worker_expectations['demand']['labor'],
            'expected_consumption_supply': self.worker_expectations['supply']['consumption']
        }
        if not check_valid_params(params):
            raise ValueError("Invalid parameters for utility maximization")
        return params

    def update_expectations(self, labor_market_stats, consumption_market_stats):
        """
        Update worker's expectations based on market statistics.
        
        :param labor_market_stats: Dictionary containing labor market statistics
        :param consumption_market_stats: Dictionary containing consumption market statistics
        """
        alpha = 1
        # Update labor market expectations
        if is_valid_number(labor_market_stats.get('price')):
            new_labor_price = alpha * labor_market_stats['price'] + (1 - alpha) * self.worker_expectations['price']['labor'][0]
            self.worker_expectations['price']['labor'] = [new_labor_price]
        if is_valid_number(labor_market_stats.get('demand')):
            new_labor_demand = alpha * labor_market_stats['demand'] + (1 - alpha) * self.worker_expectations['demand']['labor'][0]
            self.worker_expectations['demand']['labor'] = [new_labor_demand]
        
        # Update consumption market expectations
        if is_valid_number(consumption_market_stats.get('price')):
            new_consumption_price = alpha * consumption_market_stats['price'] + (1 - alpha) * self.worker_expectations['price']['consumption'][0]
            self.worker_expectations['price']['consumption'] = [new_consumption_price]
        if is_valid_number(consumption_market_stats.get('supply')):
            new_consumption_supply = alpha * consumption_market_stats['supply'] + (1 - alpha) * self.worker_expectations['supply']['consumption'][0]
            self.worker_expectations['supply']['consumption'] = [new_consumption_supply]
        if is_valid_number(consumption_market_stats.get('profit')):
            new_consumption_profit = alpha * consumption_market_stats['profit']/30 + (1 - alpha) * self.worker_expectations['profit']['consumption'][0]
            self.worker_expectations['profit']['consumption'] = [new_consumption_profit]
        
        # Update records
        self.price_record['labor'].append(self.worker_expectations['price']['labor'][0])
        self.price_record['consumption'].append(self.worker_expectations['price']['consumption'][0])
        self.demand_record['labor'].append(self.worker_expectations['demand']['labor'][0])
        self.supply_record['consumption'].append(self.worker_expectations['supply']['consumption'][0])
