from mesa import Model
from mesa.time import RandomActivationByType
from Utilities.Config import Config
from mesa_worker import Worker
from mesa_firm import Firm1, Firm2
from Utilities.mesa_market_matching import market_matching
from Utilities.Accounting_System import GlobalAccountingSystem
from Utilities.economy_data_collector import EconomyDataCollector
import numpy as np


class EconomyModel(Model):
    def __init__(self, num_workers, num_firm1, num_firm2, mode):
        super().__init__()
        self.num_workers = num_workers
        self.num_firm1 = num_firm1
        self.num_firm2 = num_firm2
        self.schedule = RandomActivationByType(self)
        self.config = Config()
        self.step_count = 0
        self.global_accounting = GlobalAccountingSystem()
        self.mode = mode
        self.data_collector = EconomyDataCollector(self)

        self.pre_labor_transactions = np.zeros(4)
        self.pre_capital_transactions = np.zeros(4)
        self.pre_consumption_transactions = np.zeros(4)
        self.labor_transactions = []
        self.labor_transactions_history = []
        self.capital_transactions = []
        self.capital_transactions_history = []
        self.consumption_transactions = []
        self.consumption_transactions_history = []
        self.create_agents()
        self.running = True

    def create_agents(self):
        # Create workers
        for i in range(self.num_workers):
            worker = Worker(i, self)
            self.schedule.add(worker)

        # Create Firm2 instances
        for i in range(self.num_firm2):
            firm = Firm2(self.num_workers + i, self)
            self.schedule.add(firm)
            self.global_accounting.register_firm(firm)

        # Create Firm1 instances
        for i in range(self.num_firm1):
            firm = Firm1(self.num_workers + self.num_firm2 + i, self)
            self.schedule.add(firm)
            self.global_accounting.register_firm(firm)

    def step(self):
        # Clear previous transactions
        self.labor_transactions.clear()
        self.capital_transactions.clear()
        self.consumption_transactions.clear()
        self.pre_labor_transactions.fill(0)
        self.pre_capital_transactions.fill(0)
        self.pre_consumption_transactions.fill(0)

        # Execute the step
        self.schedule.step()

        # Update firms
        for agent in self.schedule.agents:
            if isinstance(agent, (Firm1, Firm2)):
                agent.update_firm_state()
                agent.update_expectations()
                agent.make_production_decision()
                agent.adjust_labor()
                if isinstance(agent, Firm2):
                    agent.adjust_investment_demand()

        # Execute markets
        self.execute_labor_market()
        self.execute_capital_market()
        self.execute_consumption_market()

        # Adjust production and prices
        for agent in self.schedule.agents:
            if isinstance(agent, Firm1):
                agent.adjust_production()
                agent.adjust_price()
            elif isinstance(agent, Firm2):
                agent.adjust_production()
                agent.adjust_price()

        # Collect data
        self.data_collector.datacollector.collect(self)
        print(f"Step {self.step_count} completed")
        self.step_count += 1


    def execute_labor_market(self):

        buyers = [(firm.labor_demand, firm.get_desired_wage(), firm, firm.get_max_wage())
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
        avg_buyer_max = sum(b[3] for b in buyers)/ len(buyers) if buyers else 0
        avg_seller_min = sum(s[3] for s in sellers)/ len(sellers) if sellers else 0
        self.pre_capital_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price, avg_buyer_max, avg_seller_min])

        transactions = market_matching(buyers, sellers)
        self.capital_transactions = transactions
        self.captial_transactions_history = np.array([len(transactions), sum(t[2] for t in transactions), sum(t[3] for t in transactions)])
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
        avg_buyer_max = sum(b[3] for b in buyers)/ len(buyers) if buyers else 0
        avg_seller_min = sum(s[3] for s in sellers)/ len(sellers) if sellers else 0
        self.pre_consumption_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price, avg_buyer_max, avg_seller_min])
        transactions = market_matching(buyers, sellers)

        self.consumption_transactions = transactions
        self.consumption_transactions_history = np.array([len(transactions), sum(t[2] for t in transactions), sum(t[3] for t in transactions)])
        for buyer, seller, quantity, price in transactions:
            buyer.consume(quantity, price)
            seller.sell_goods(quantity, price)
