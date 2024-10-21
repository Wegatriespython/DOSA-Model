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
                'labor': [30],
                'consumption': [30]
            },
            'price': {
                'labor': [1],
                'consumption': [1]
            },
            'supply': {
                'labor': [30],
                'consumption': [30]
            }
        }
        self.demand_record = {'labor': [], 'consumption': []}
        self.price_record = {'labor': [], 'consumption': []}
        self.supply_record = {'labor': [], 'consumption': []}

    def get_max_consumption_price(self):
        # set prices based on wage expectations
        savings = self.savings
        desired_consumption = self.desired_consumption
        income = self.worker_expectations['price']['labor'][0] * self.working_hours
        price = (savings + income) / desired_consumption if desired_consumption != 0 else 0
        return price

    def get_min_wage(self):
        # Set wages based on consumption price expectations
        expenses = self.desired_consumption * self.worker_expectations['price']['consumption'][0]
        wage = expenses / self.working_hours if self.working_hours != 0 else expenses
        return wage


    def get_utility_params(self):
        params = {
            'savings': 2,
            'wage': self.worker_expectations['price']['labor'],
            'price': self.worker_expectations['price']['consumption'],
            'discount_rate': 0.05,
            'time_horizon': self.time_horizon,
            'alpha': 0.9,
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
        # Update labor market expectations
        if is_valid_number(labor_market_stats.get('price')):
            self.worker_expectations['price']['labor'] = [labor_market_stats['price']]
        if is_valid_number(labor_market_stats.get('demand')):
            self.worker_expectations['demand']['labor'] = [labor_market_stats['demand']]
        
        # Update consumption market expectations
        if is_valid_number(consumption_market_stats.get('price')):
            self.worker_expectations['price']['consumption'] = [consumption_market_stats['price']]
        if is_valid_number(consumption_market_stats.get('supply')):
            self.worker_expectations['supply']['consumption'] = [consumption_market_stats['supply']]
        
        # Update records
        self.price_record['labor'].append(self.worker_expectations['price']['labor'][0])
        self.price_record['consumption'].append(self.worker_expectations['price']['consumption'][0])
        self.demand_record['labor'].append(self.worker_expectations['demand']['labor'][0])
        self.supply_record['consumption'].append(self.worker_expectations['supply']['consumption'][0])
