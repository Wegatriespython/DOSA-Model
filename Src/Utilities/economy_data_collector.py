from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
from mesa_worker import Worker
from mesa_firm import Firm1, Firm2
class EconomyDataCollector:
    def __init__(self, model):
        self.model = model
        self.data_collection = {}
        self.datacollector = DataCollector(
            model_reporters={
                "Total Labor": self.get_total_labor,
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
                "capital_raw_demand": lambda m: m.pre_capital_transactions[0] if len(m.pre_capital_transactions) > 0 else 0,
                "capital_raw_supply": lambda m: m.pre_capital_transactions[1] if len(m.pre_capital_transactions) > 1 else 0,
                "capital_raw_buyer_price": lambda m: m.pre_capital_transactions[2] if len(m.pre_capital_transactions) > 2 else 0,
                "capital_raw_seller_price": lambda m: m.pre_capital_transactions[3] if len(m.pre_capital_transactions) > 3 else 0,
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

            },
            agent_reporters={
                "Type": lambda a: type(a).__name__,
                "Capital": lambda a: getattr(a, 'capital', None),
                "Labor": lambda a: len(getattr(a, 'workers', [])),
                "Working_Hours": lambda a: getattr(a, 'total_working_hours', None),
                "Labor_Demand": lambda a: getattr(a, 'total_working_hours', None),
                "Production": lambda a: getattr(a, 'production', None),
                "Optimals": lambda a: getattr(a, 'optimals', None),
                "Expectations": lambda a: getattr(a, 'expectations', None),
                "Investment": lambda a: getattr(a, 'investment_demand', None),
                "Sales": lambda a: getattr(a, 'sales', None),
                "Wages_Firm": lambda a: getattr(a, 'wage', None),
                "Price": lambda a: getattr(a, 'price', None),
                "Inventory": lambda a: getattr(a, 'inventory', None),
                "Budget": lambda a: getattr(a, 'budget', None),
                "Productivity": lambda a: getattr(a, 'productivity', None),
                "Wage": lambda a: getattr(a, 'wage', None),
                "Income": lambda a: getattr(a, 'income', None),
                "Per_Worker_Income": lambda a: getattr(a, 'per_worker_income',None),
                "Skills": lambda a: getattr(a, 'skills', None),
                "Savings": lambda a: getattr(a, 'savings', None),
                "Consumption": lambda a: getattr(a, 'consumption', None),
                "desired_consumption": lambda a: getattr(a, 'desired_consumption', None)
            }
        )
    @staticmethod
    def get_total_labor(model):
        return sum(worker.total_working_hours for worker in model.schedule.agents if isinstance(worker, Worker))
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
        demands = [firm.expected_demand for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2))]
        return sum(demands) / len(demands) if demands else 0
    @staticmethod
    def get_average_consumption_demand(model):
        demands = [agent.desired_consumption for agent in model.schedule.agents if isinstance(agent, Worker)]
        return sum(demands)/5  if demands else 0
    @staticmethod
    def get_average_consumption_demand_expected(model):
        demands = [firm.expected_demand for firm in model.schedule.agents if isinstance(firm,  Firm2)]
        return sum(demands) / len(demands) if demands else 0
    @staticmethod
    def average_sales(model):
        sales = [firm.sales for firm in model.schedule.agents if isinstance(firm, (Firm2, Firm1))]
        return sum(sales) / len(sales) if sales else 0
    @staticmethod
    def get_average_capital_price(model):
        prices = [firm.price for firm in model.schedule.agents if isinstance(firm, Firm1)]
        return sum(prices) / len(prices) if prices else 3

    @staticmethod
    def get_average_wage(model):
        if model.step_count == 0:
            return model.config.MINIMUM_WAGE
        employed_workers = [worker for worker in model.schedule.agents if isinstance(worker, Worker) and worker.total_working_hours > 0]
        if employed_workers:
            return sum(worker.wage for worker in employed_workers) / len(employed_workers)
        return model.config.MINIMUM_WAGE
    @staticmethod
    def get_average_consumption_expected_price(model):
        prices = [firm.expected_price for firm in model.schedule.agents if isinstance(firm, Firm2)]
        if prices:
            return sum(prices) / len(prices)
        return 0
    @staticmethod
    def get_average_consumption_good_price(model):
        prices = [firm.price for firm in model.schedule.agents if isinstance(firm, Firm2)]
        if prices:
            return sum(prices) / len(prices)
        return 0

    @staticmethod
    def get_total_demand(model):
        capital_demand= sum(firm.investment_demand for firm in model.schedule.agents if isinstance(firm,  Firm2))
        consumption_demand = sum(worker.desired_consumption for worker in model.schedule.agents if isinstance(worker, Worker))
        return capital_demand + consumption_demand
    @staticmethod
    def get_total_sales(model):
        return sum(firm.sales for firm in model.schedule.agents if isinstance(firm, (Firm1, Firm2)))
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
        total_labor_hours = EconomyDataCollector.get_total_labor(model)
        return total_output / total_labor_hours if total_labor_hours > 0 else 0
