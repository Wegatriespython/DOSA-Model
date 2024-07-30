from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
import logging
from Config import Config
from mesa_worker import Worker
from mesa_firm import Firm1, Firm2
from mesa_market_matching import market_matching
from Accounting_System import GlobalAccountingSystem

class EconomyModel(Model):
    def __init__(self, num_workers, num_firm1, num_firm2):
        super().__init__()
        self.num_workers = num_workers
        self.num_firm1 = num_firm1
        self.num_firm2 = num_firm2
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(10, 10, True)
        self.config = Config()
        self.step_count = 0
        self.global_accounting = GlobalAccountingSystem()
        
        self.datacollector = DataCollector(
            model_reporters={
                "Total Labor": lambda m: m.global_accounting.total_labor,
                "Total Capital": lambda m: m.global_accounting.total_capital,
                "Total Goods": lambda m: m.global_accounting.total_goods,
                "Total Money": lambda m: m.global_accounting.total_money,
                "Average Market Demand": lambda m: m.global_accounting.get_average_market_demand(),
                "Average Capital Price": lambda m: m.global_accounting.get_average_capital_price(),
                "Average Wage": lambda m: m.global_accounting.get_average_wage(),
                "Average Inventory": self.calculate_average_inventory, 
                "Average Consumption Good Price": lambda m: m.global_accounting.get_average_consumption_good_price(),
                "Total Demand": lambda m: m.global_accounting.get_total_demand(),
                "Total Production": lambda m: m.global_accounting.get_total_production(),
                "Global Productivity": self.calculate_global_productivity,
            },
            agent_reporters={
                "Type": lambda a: type(a).__name__,
                "Capital": lambda a: getattr(a, 'capital', None),
                "Labor": lambda a: len(getattr(a, 'workers', [])),
                "Production": lambda a: getattr(a, 'production', None),
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

        self.create_agents()
        self.running = True

    def create_agents(self):
        # Create workers
        for i in range(self.num_workers):
            worker = Worker(i, self)
            self.schedule.add(worker)
            x, y = self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)
            self.grid.place_agent(worker, (x, y))

        # Create Firm1 instances
        for i in range(self.num_firm1):
            firm = Firm1(self.num_workers + i, self)
            self.schedule.add(firm)
            self.global_accounting.register_firm(firm)
            x, y = self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)
            self.grid.place_agent(firm, (x, y))

        # Create Firm2 instances
        for i in range(self.num_firm2):
            firm = Firm2(self.num_workers + self.num_firm1 + i, self)
            self.schedule.add(firm)
            self.global_accounting.register_firm(firm)
            x, y = self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)
            self.grid.place_agent(firm, (x, y))

    def step(self):
        self.update_worker_price_information()  # Add this line
        self.schedule.step()
        self.execute_markets()
        self.update_global_accounting()
        self.datacollector.collect(self)
        self.global_accounting.reset_period_data()
        self.step_count += 1
    def update_worker_price_information(self):
        consumption_firms = [firm for firm in self.schedule.agents if isinstance(firm, Firm2)]
        seller_prices = [firm.price for firm in consumption_firms]
        
        for agent in self.schedule.agents:
            if isinstance(agent, Worker):
                agent.set_seller_prices(seller_prices)  
    def execute_markets(self):
        for agent in self.schedule.agents:
            if isinstance(agent, (Firm1, Firm2)):
                agent.update_firm_state()
        self.execute_labor_market()
        self.execute_capital_market()
        self.execute_consumption_market()

    def execute_labor_market(self):
        print("Labor Market")
        buyers = [(firm.labor_demand, firm.get_max_wage(), firm) 
                  for firm in self.schedule.agents 
                  if isinstance(firm, (Firm1, Firm2)) and firm.labor_demand > 0]
        sellers = [(1, worker.wage, worker) 
                   for worker in self.schedule.agents 
                   if isinstance(worker, Worker) and not worker.employed]
        
        transactions = market_matching(buyers, sellers)
        for firm, worker, quantity, price in transactions:
            firm.hire_worker(worker, price)
            worker.get_hired(firm, price)
            self.global_accounting.record_labor_transaction(firm, worker, quantity, price)

    def execute_capital_market(self):
        print("Capital Market")
        buyers = [(firm.investment_demand, firm.get_max_capital_price(), firm) 
                  for firm in self.schedule.agents 
                  if isinstance(firm, Firm2) and firm.investment_demand > 0]
        sellers = [(firm.inventory, firm.price, firm) 
                   for firm in self.schedule.agents 
                   if isinstance(firm, Firm1) and firm.inventory > 0]
        
        transactions = market_matching(buyers, sellers)
        for buyer, seller, quantity, price in transactions:
            buyer.buy_capital(quantity, price)
            seller.sell_goods(quantity, price)
            self.global_accounting.record_capital_transaction(buyer, seller, quantity, price)

    def execute_consumption_market(self):
        print("Consumption Market")
        buyers = [(worker.consumption, worker.get_max_consumption_price(), worker) 
                  for worker in self.schedule.agents 
                  if isinstance(worker, Worker) and worker.savings > 0]
        sellers = [(firm.inventory, firm.price, firm) 
                   for firm in self.schedule.agents 
                   if isinstance(firm, Firm2) and firm.inventory > 0]
        
        transactions = market_matching(buyers, sellers)
        for buyer, seller, quantity, price in transactions:
            buyer.consume(quantity, price)
            seller.sell_goods(quantity, price)
            self.global_accounting.record_consumption_transaction(buyer, seller, quantity, price)

    def update_global_accounting(self):
        total_demand = sum(firm.expected_demand for firm in self.global_accounting.firms)
        self.global_accounting.update_market_demand(total_demand)

    def calculate_average_inventory(self):
        firms = [agent for agent in self.schedule.agents if isinstance(agent, (Firm1, Firm2))]
        return sum(firm.inventory for firm in firms) / len(firms) if firms else 0

    def calculate_global_productivity(self):
        total_output = sum(firm.production for firm in self.global_accounting.firms)
        total_labor = self.global_accounting.total_labor
        return total_output / total_labor if total_labor > 0 else 0