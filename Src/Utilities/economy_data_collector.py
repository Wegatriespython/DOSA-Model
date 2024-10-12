from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
from worker import Worker
from firm import Firm1, Firm2
class EconomyDataCollector:
    def __init__(self, model):
        self.model = model
        self.data_collection = {}
        self.datacollector = DataCollector(
            model_reporters={
                "Total Labor Supply": self.get_total_labor_supply,
                "Total Capital": self.get_total_capital,
                "Total Goods": self.get_total_goods,
                "Total Money": self.get_total_money,
                "Average Market Demand": self.get_average_market_demand,
                "Average Capital Price": self.get_average_capital_price,
                "Average_Consumption_Demand": self.get_average_consumption_demand,
                "Average_Consumption_Demand_Expected": self.get_average_consumption_demand_expected,
                "Average_Consumption_expected_price": self.get_average_consumption_expected_price,
                "Average Wage": self.get_average_wage,
                "Average Inventory": self.calculate_average_inventory,
                "Average Consumption Good Price": self.get_average_consumption_good_price,
                "Total Demand": self.get_total_demand,
                "Total Supply": self.get_total_sales,
                "Total Production": self.get_total_production,
                "Global Productivity": self.calculate_global_productivity,
                "labor_raw_demand": lambda m: m.pre_labor_transactions[0] if len(m.pre_labor_transactions) > 0 else 0,
                "labor_raw_supply": lambda m: m.pre_labor_transactions[1] if len(m.pre_labor_transactions) > 1 else 0,
                "labor_raw_buyer_price": lambda m: m.pre_labor_transactions[2] if len(m.pre_labor_transactions) > 2 else 0,
                "labor_raw_seller_price": lambda m: m.pre_labor_transactions[3] if len(m.pre_labor_transactions) > 3 else 0,
                "labor_raw_buyer_max": lambda m: m.pre_labor_transactions[4] if len(m.pre_labor_transactions) > 4 else 0,
                "labor_raw_seller_min": lambda m: m.pre_labor_transactions[5] if len(m.pre_labor_transactions) > 5 else 0,
                "capital_raw_demand": lambda m: m.pre_capital_transactions[0] if len(m.pre_capital_transactions) > 0 else 0,
                "capital_raw_supply": lambda m: m.pre_capital_transactions[1] if len(m.pre_capital_transactions) > 1 else 0,
                "capital_raw_buyer_price": lambda m: m.pre_capital_transactions[2] if len(m.pre_capital_transactions) > 2 else 0,
                "capital_raw_seller_price": lambda m: m.pre_capital_transactions[3] if len(m.pre_capital_transactions) > 3 else 0,
                "capital_raw_buyer_max": lambda m: m.pre_capital_transactions[4] if len(m.pre_capital_transactions) > 4 else 0,
                "capital_raw_seller_min": lambda m: m.pre_capital_transactions[5] if len(m.pre_capital_transactions) > 5 else 0,
                "consumption_raw_demand": lambda m: m.pre_consumption_transactions[0] if len(m.pre_consumption_transactions) > 0 else 0,
                "consumption_raw_supply": lambda m: m.pre_consumption_transactions[1] if len(m.pre_consumption_transactions) > 1 else 0,
                "consumption_raw_buyer_price": lambda m: m.pre_consumption_transactions[2] if len(m.pre_consumption_transactions) > 2 else 0,
                "consumption_raw_seller_price": lambda m: m.pre_consumption_transactions[3] if len(m.pre_consumption_transactions) > 3 else 0,
                "consumption_raw_buyer_max": lambda m: m.pre_consumption_transactions[4] if len(m.pre_consumption_transactions)>3 else 0,
                "consumption_raw_seller_min": lambda m:m.pre_consumption_transactions[5] if len
                (m.pre_consumption_transactions)>3 else 0,
                "labor_Market_Quantity": lambda m: sum(t[2] for t in m.labor_transactions),
                "labor_Market_Price": lambda m: np.mean([t[3] for t in m.labor_transactions]) if m.labor_transactions else 0,
                "capital_Market_Quantity": lambda m: sum(t[2] for t in m.capital_transactions),
                "capital_Market_Price": lambda m: np.mean([t[3] for t in m.capital_transactions]) if m.capital_transactions else 0,
                "consumption_Market_Quantity": lambda m: sum(t[2] for t in m.consumption_transactions),
                "consumption_Market_Price": lambda m: np.mean([t[3] for t in m.consumption_transactions]) if m.consumption_transactions else 0,
                "Average_Firm_Expectations_Demand": self.get_average_firm_expectations_demand,
                "Average_Firm_Expectations_Price": self.get_average_firm_expectations_price,
                "Average_Firm_Expectations_Supply": self.get_average_firm_expectations_supply,
                "Average_Worker_Expectations_Demand": self.get_average_worker_expectations_demand,
                "Average_Worker_Expectations_Price": self.get_average_worker_expectations_price,
                "Average_Worker_Expectations_Supply": self.get_average_worker_expectations_supply,
            },
            agent_reporters={
                "Type": lambda a: type(a).__name__,
                "Capital": lambda a: getattr(a, 'capital', None),
                "Labor": lambda a: len(getattr(a, 'workers', [])),
                "Total_Working_Hours": lambda a: getattr(a, 'total_working_hours', None),
                "Working_Hours": lambda a: getattr(a, 'working_hours', None),
                "Labor_Demand": lambda a: getattr(a, 'total_working_hours', None),
                "desired_consumption": lambda a: getattr(a, 'desired_consumption', None),
                "Production": lambda a: getattr(a, 'production', None),
                "Optimals": lambda a: getattr(a, 'optimals', None),
                "Firm_Expectations_Demand": lambda a: self.format_expectations(a, 'firm_expectations', 'demand'),
                "Firm_Expectations_Price": lambda a: self.format_expectations(a, 'firm_expectations', 'price'),
                "Firm_Expectations_Supply": lambda a: self.format_expectations(a, 'firm_expectations', 'supply'),
                "Worker_Expectations_Demand": lambda a: self.format_expectations(a, 'worker_expectations', 'demand'),
                "Worker_Expectations_Price": lambda a: self.format_expectations(a, 'worker_expectations', 'price'),
                "Worker_Expectations_Supply": lambda a: self.format_expectations(a, 'worker_expectations', 'supply'),
                "Demand_Record": lambda a: getattr(a, 'demand_record', {}),
                "Price_Record": lambda a: getattr(a, 'price_record', {}),
                "Supply_Record": lambda a: getattr(a, 'supply_record', {}),
                "Performance_Record": lambda a: getattr(a, 'performance_record', {}),
                "Gaps_Record": lambda a: getattr(a, 'gaps_record', {}),
                "Investment": lambda a: getattr(a, 'investment_demand', None),
                "Sales": lambda a: getattr(a, 'sales', None),
                "Wages_Firm": lambda a: getattr(a, 'wage', None),
                "Price": lambda a: getattr(a, 'price', None),
                "Inventory": lambda a: getattr(a, 'inventory', None),
                "Budget": lambda a: getattr(a, 'budget', None),
                "Productivity": lambda a: getattr(a, 'productivity', None),
                "Wage": lambda a: getattr(a, 'wage', None),
                "Income": lambda a: getattr(a, 'income', None),
                "Per_Worker_Income": lambda a: getattr(a, 'per_worker_income', None),
                "Skills": lambda a: getattr(a, 'skills', None),
                "Savings": lambda a: getattr(a, 'savings', None),
                "Consumption": lambda a: getattr(a, 'consumption', None)

            }
        )
    @staticmethod
    def get_total_labor_supply(model):
        return sum(worker.available_hours() for worker in model.schedule.agents if isinstance(worker, Worker))
    @staticmethod
    def get_total_labor_demand(model):
        return sum(firm.labor_demand for firm in model.schedule.agents if isinstance(firm, (Firm1,Firm2)))
    @staticmethod
    def get_total_capital(model):
        return sum(firm.capital for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2)))

    @staticmethod
    def get_total_goods(model):
        return sum(firm.inventory for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2)))

    @staticmethod
    def get_total_money(model):
        return sum(firm.budget for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2))) + \
                sum(worker.savings for worker in model.schedule.agents if isinstance(worker, Worker))
    @staticmethod
    def get_average_market_demand(model):
        return (model.pre_labor_transactions[0] + model.pre_capital_transactions[0] + model.pre_consumption_transactions[0]) / 3

    @staticmethod
    def get_average_capital_price(model):
        return model.pre_capital_transactions[2] if len(model.pre_capital_transactions) > 2 else 0

    @staticmethod
    def get_average_consumption_demand(model):
        return model.pre_consumption_transactions[0] if len(model.pre_consumption_transactions) > 0 else 0

    @staticmethod
    def get_average_consumption_demand_expected(model):
        return model.pre_consumption_transactions[0] if len(model.pre_consumption_transactions) > 0 else 0

    @staticmethod
    def get_average_consumption_expected_price(model):
        return model.pre_consumption_transactions[2] if len(model.pre_consumption_transactions) > 2 else 0

    @staticmethod
    def get_average_consumption_good_price(model):
        return np.mean([t[3] for t in model.consumption_transactions]) if model.consumption_transactions else 0
    @staticmethod
    def get_average_wage(model):
      return np.mean([t[3] for t in model.labor_transactions]) if model.labor_transactions else 0
    @staticmethod
    def get_total_demand(model):
        return model.pre_labor_transactions[0] + model.pre_capital_transactions[0] + model.pre_consumption_transactions[0]

    @staticmethod
    def get_total_sales(model):
        return sum(t[2] for t in model.labor_transactions) + sum(t[2] for t in model.capital_transactions) + sum(t[2] for t in model.consumption_transactions)
    @staticmethod
    def get_total_production(model):
        return sum(firm.production for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2)))

    @staticmethod
    def employment_snapshot(model):
       if model.step_count % 10 == 0:
           for firm in model.schedule.agents:
                if isinstance(firm, (Firm1,Firm2)):
                    print(firm.unique_id, firm.workers)

    def export_data_to_csv(self):
        all_data = []
        for firm_id, firm_data in self.data_collection.items():
            for features, actual_demand, sales in firm_data:
                row = list(features) + [actual_demand, sales, firm_id]
                all_data.append(row)

        columns = [
            'capital', 'num_workers', 'productivity', 'price', 'inventory', 'budget',
            'mean_historic_sales', 'std_historic_sales', 'avg_wage', 'avg_capital_price',
            'avg_consumption_good_price', 'market_demand', 'actual_demand', 'sales', 'firm_id'
        ]

        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv('economic_model_data.csv', index=False)
        print("Data exported to economic_model_data.csv")

    def get_current_state(self):
        # New method to provide current state to central planner
        return {
            'workers': self.workers,
            'firms1': self.firm1s,
            'firms2': self.firm2s,
            'relative_price': self.relative_price,
            'total_labor': self.global_accounting.total_labor,
            'total_capital': self.global_accounting.total_capital,
            'total_goods': self.global_accounting.total_goods,
            'total_money': self.global_accounting.total_money,
        }

    @staticmethod
    def calculate_average_inventory(model):
        firms = [agent for agent in model.schedule.agents if isinstance(agent, (Firm1, Firm2))]
        return sum(firm.inventory for firm in firms) / len(firms) if firms else 0

    @staticmethod
    def calculate_global_productivity(model):
        total_output = sum(firm.production for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2)))
        total_labor_hours = EconomyDataCollector.get_total_labor_supply(model)
        return total_output / total_labor_hours if total_labor_hours > 0 else 0

    @staticmethod
    def format_expectations(agent, expectation_type, category):
        expectations = getattr(agent, expectation_type, {}).get(category, {})
        return {k: v[0] if isinstance(v, list) and v else v for k, v in expectations.items()}

    def get_average_expectations(self, agent_types, expectation_type, expectation_category):
        expectations = [getattr(agent, expectation_type) for agent in self.model.schedule.agents if isinstance(agent, agent_types)]
        return {
            k: np.mean([
                exp.get(k, [0])[0] if isinstance(exp.get(k, []), list) and exp.get(k, []) else exp.get(k, 0) 
                for exp in expectations
            ]) 
            for k in ['consumption', 'capital', 'labor']
        }

    def get_average_firm_expectations_demand(self):
        return self.get_average_expectations((Firm1, Firm2), 'firm_expectations', 'demand')

    def get_average_firm_expectations_price(self):
        return self.get_average_expectations((Firm1, Firm2), 'firm_expectations', 'price')

    def get_average_firm_expectations_supply(self):
        return self.get_average_expectations((Firm1, Firm2), 'firm_expectations', 'supply')

    def get_average_worker_expectations_demand(self):
        return self.get_average_expectations(Worker, 'worker_expectations', 'demand')

    def get_average_worker_expectations_price(self):
        return self.get_average_expectations(Worker, 'worker_expectations', 'price')

    def get_average_worker_expectations_supply(self):
        return self.get_average_expectations(Worker, 'worker_expectations', 'supply')