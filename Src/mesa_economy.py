from mesa import Model
from mesa.time import RandomActivationByType, SimultaneousActivation
from mesa.datacollection import DataCollector
import logging
from numpy.lib.function_base import average
import pandas as pd
import random
import numpy as np
from pandas.compat import numpy
from Utilities.Config import Config
from mesa_worker import Worker
from mesa_firm import Firm1, Firm2
from Utilities.mesa_market_matching import market_matching
from Utilities.Accounting_System import GlobalAccountingSystem
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

class EconomyModel(Model):
    def __init__(self, num_workers, num_firm1, num_firm2, mode):
        super().__init__()
        self.num_workers = num_workers
        self.num_firm1 = num_firm1
        self.num_firm2 = num_firm2
        self.schedule = RandomActivationByType(self)
        self.config = Config()
        self.step_count = 0
        self.global_accounting = GlobalAccountingSystem()  # Keep this for background data collection
        self.mode = mode
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
                "Skills": lambda a: getattr(a, 'skills', None),
                "Savings": lambda a: getattr(a, 'savings', None),
                "Consumption": lambda a: getattr(a, 'consumption', None)
            }
        )
        # Initialize numpy arrays with zeros
        self.pre_labor_transactions = np.zeros(4)
        self.pre_capital_transactions = np.zeros(4)
        self.pre_consumption_transactions = np.zeros(4)
        self.labor_transactions = []
        self.capital_transactions = []
        self.consumption_transactions = []
        self.create_agents()
        self.running = True
    def get_total_labor(self):
        return sum(worker.total_working_hours for worker in self.schedule.agents if isinstance(worker, Worker))
    def get_total_labor_demand(self):
        return sum(firm.labor_demand for firm in self.schedule.agents if isinstance(firm, (Firm1,Firm2)))
    def get_total_capital(self):
        return sum(firm.capital for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2)))

    def get_total_goods(self):
        return sum(firm.inventory for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2)))

    def get_total_money(self):
        return sum(firm.budget for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2))) + \
                sum(worker.savings for worker in self.schedule.agents if isinstance(worker, Worker))

    def get_average_market_demand(self):
        demands = [firm.expected_demand for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2))]
        return sum(demands) / len(demands) if demands else 0
    def get_average_consumption_demand(self):
        demands = [agent.desired_consumption for agent in self.schedule.agents if isinstance(agent, Worker)]
        return sum(demands)/5  if demands else 0
    def get_average_consumption_demand_expected(self):
        demands = [firm.expected_demand for firm in self.schedule.agents if isinstance(firm,  Firm2)]
        return sum(demands) / len(demands) if demands else 0
    def average_sales(self):
        sales = [firm.sales for firm in self.schedule.agents if isinstance(firm, (Firm2, Firm1))]
        return sum(sales) / len(sales) if sales else 0
    def get_average_capital_price(self):
        prices = [firm.price for firm in self.schedule.agents if isinstance(firm, Firm1)]
        return sum(prices) / len(prices) if prices else 3

    def get_average_wage(self):
        if self.step_count == 0:
            return self.config.MINIMUM_WAGE
        employed_workers = [worker for worker in self.schedule.agents if isinstance(worker, Worker) and worker.total_working_hours > 0]
        if employed_workers:
            return sum(worker.wage for worker in employed_workers) / len(employed_workers)
        return self.config.MINIMUM_WAGE
    def get_average_consumption_expected_price(self):
        prices = [firm.expected_price for firm in self.schedule.agents if isinstance(firm, Firm2)]
        if prices:
            return sum(prices) / len(prices)
        return 0
    def get_average_consumption_good_price(self):
        prices = [firm.price for firm in self.schedule.agents if isinstance(firm, Firm2)]
        if prices:
            return sum(prices) / len(prices)
        return 0

    def get_total_demand(self):
        capital_demand= sum(firm.investment_demand for firm in self.schedule.agents if isinstance(firm,  Firm2))
        consumption_demand = sum(worker.desired_consumption for worker in self.schedule.agents if isinstance(worker, Worker))
        return capital_demand + consumption_demand
    def get_total_sales(self):
        return sum(firm.sales for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2)))
    def get_total_production(self):
        return sum(firm.production for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2)))
    def create_agents(self):
        # Create and add workers first
        for i in range(self.num_workers):
            worker = Worker(i, self)
            self.schedule.add(worker)


        # Create and add Firm2 instances second
        for i in range(self.num_firm2):
            firm = Firm2(self.num_workers + i, self)
            self.schedule.add(firm)
            self.global_accounting.register_firm(firm)
            self.data_collection[firm.unique_id] = []


        # Create and add Firm1 instances last
        for i in range(self.num_firm1):
            firm = Firm1(self.num_workers + self.num_firm2 + i, self)
            self.schedule.add(firm)
            self.global_accounting.register_firm(firm)
            self.data_collection[firm.unique_id] = []



    def step(self):
        self.labor_transactions.clear()
        self.capital_transactions.clear()
        self.consumption_transactions.clear()
        self.pre_labor_transactions.fill(0)
        self.pre_capital_transactions.fill(0)
        self.pre_consumption_transactions.fill(0)
        self.schedule.step()
        for agent in self.schedule.agents:
            if isinstance(agent, (Firm1, Firm2)):
                agent.update_firm_state()
                agent.get_expected_demand()
                agent.make_production_decision()

                agent.adjust_labor()
                if isinstance(agent, Firm2):
                    agent.adjust_investment_demand()
        self.execute_labor_market()

        for agent in self.schedule.agents:
            if isinstance(agent, Firm1):
                agent.adjust_production()
                print(agent.unique_id, agent.production)
        self.execute_capital_market()
        for agent in self.schedule.agents:
            if isinstance(agent, Firm1):
                agent.adjust_price()
        for agent in self.schedule.agents:
            if isinstance(agent, Firm2):
                agent.adjust_production()
        self.execute_consumption_market()
        for agent in self.schedule.agents:
            if isinstance(agent, Firm2):
                agent.adjust_price()
        self.employment_snapshot()
        #self.collect_data()
        self.datacollector.collect(self)
        print(f"Step {self.step_count} completed")
        self.step_count += 1


    def employment_snapshot(self):
       if self.step_count % 10 == 0:
           for firm in self.schedule.agents:
                if isinstance(firm, (Firm1,Firm2)):
                    print(firm.unique_id, firm.workers)

    def collect_data(self):
        for firm in self.schedule.agents:
            if isinstance(firm, (Firm1, Firm2)):
                features = firm.prepare_features()
                actual_demand = firm.get_market_demand(firm.get_market_type())
                target = firm.sales
                if firm.unique_id not in self.data_collection:
                    self.data_collection[firm.unique_id] = []
                self.data_collection[firm.unique_id].append((features, actual_demand, target))
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

    def execute_labor_market(self):

        buyers = [(firm.labor_demand, firm.calculate_average_wage(), firm, firm.get_max_wage())
                  for firm in self.schedule.agents
                  if isinstance(firm, (Firm1, Firm2)) and firm.labor_demand > 0]

        sellers = [(worker.available_hours(), worker.expected_wage, worker, worker.get_min_wage())
                   for worker in self.schedule.agents
                   if isinstance(worker, Worker) and worker.available_hours() > 0]
        transactions = market_matching(buyers, sellers)
        #print("Labor Market Transaction", transactions)
        buyer_demand = sum(b[0] for b in buyers) if buyers else 0
        seller_inventory = sum(s[0] for s in sellers) if sellers else 0
        avg_buyer_price = sum(b[1] for b in buyers) / len(buyers) if buyers else 0
        avg_seller_price = sum(s[1] for s in sellers) / len(sellers) if sellers else 0

        self.pre_labor_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price])
        self.labor_transactions = transactions
        for firm, worker, hours, price in transactions:
            # Round hours to the nearest integer
            hours = round(hours)
            if hours > 0:
                firm.hire_worker(worker, price, hours)

        # Update labor demand for firms
        for firm in self.schedule.agents:
            if isinstance(firm, (Firm1, Firm2)):
                firm.labor_demand = max(0, firm.labor_demand - sum(t[2] for t in transactions if t[0] == firm))

    def execute_capital_market(self):

        buyers = [(firm.investment_demand, firm.get_desired_capital_price(), firm, firm.get_max_capital_price())
                  for firm in self.schedule.agents
                  if isinstance(firm, Firm2) and firm.investment_demand > 0]
        print(buyers)

        sellers = []
        for firm in self.schedule.agents:
            match firm:
                case Firm1() if firm.inventory > 0:
                    sellers.append((firm.inventory, firm.price, firm, firm.get_min_sale_price()))
                case Firm2() if firm.capital_inventory > 0:
                    sellers.append((firm.capital_inventory,firm.capital_resale_price, firm, 0.1))
        buyer_demand = sum(b[0] for b in buyers)  if buyers else 0
        seller_inventory = sum(s[0] for s in sellers)  if sellers else 0
        avg_buyer_price = sum(b[1] for b in buyers) / len(buyers) if buyers else 0
        avg_seller_price = sum(s[1] for s in sellers) / len(sellers) if sellers else 0

        self.pre_capital_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price])
        transactions = market_matching(buyers, sellers)
        self.capital_transactions = transactions
        for buyer, seller, quantity, price in transactions:
            buyer.buy_capital(quantity, price)
            seller.sell_goods(quantity, price)
            # Remove global_accounting.record_capital_transaction


    def execute_consumption_market(self):
        #print("Executing consumption market")
        buyers = [(worker.desired_consumption, worker.expected_price, worker, worker.get_max_consumption_price())
                  for worker in self.schedule.agents
                  if isinstance(worker, Worker) and worker.savings > 0]
        #print("Buyers", buyers)
        sellers = [(firm.inventory, firm.price, firm, firm.get_min_sale_price())
                   for firm in self.schedule.agents
                   if isinstance(firm, Firm2) and firm.inventory > 0]
        if sellers:
            min_price = max(sellers, key=lambda x: x[1])[1]
            max_inventory = max(sellers, key=lambda x: x[0])[0]
            #print("Min price, max inventory", min_price, max_inventory)
        #print("Sellers", sellers)


        buyer_demand = sum(b[0] for b in buyers) if buyers else 0
        seller_inventory = sum(s[0] for s in sellers) if sellers else 0
        avg_buyer_price = sum(b[1] for b in buyers) / len(buyers) if buyers else 0
        avg_seller_price = sum(s[1] for s in sellers) / len(sellers) if sellers else 0

        self.pre_consumption_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price])
        transactions = market_matching(buyers, sellers)

        self.consumption_transactions = transactions
        #print("Consumption Market Transaction", transactions)
        for buyer, seller, quantity, price in transactions:
            buyer.consume(quantity, price)
            seller.sell_goods(quantity, price)

    def calculate_average_inventory(self):
        firms = [agent for agent in self.schedule.agents if isinstance(agent, (Firm1, Firm2))]
        return sum(firm.inventory for firm in firms) / len(firms) if firms else 0

    def calculate_global_productivity(self):
        total_output = sum(firm.production for firm in self.schedule.agents if isinstance(firm, (Firm1, Firm2)))
        total_labor_hours = self.get_total_labor()
        return total_output / total_labor_hours if total_labor_hours > 0 else 0
