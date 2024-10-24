from mesa import Model
from mesa.time import RandomActivationByType
from Utilities.Config import Config
from worker import Worker
from firm import Firm2, Firm1
from Utilities.market_matching import market_matching
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
        self.mode = mode
        self.data_collector = EconomyDataCollector(self)

        self.pre_labor_transactions = np.zeros(4)
        self.pre_capital_transactions = np.zeros(4)
        self.pre_consumption_transactions = np.zeros(4)
        self.labor_supply = 0
        self.capital_supply = 0
        self.consumption_supply = 0
        self.capital_supply_history = []
        self.labor_supply_history = []
        self.consumption_supply_history = []
        self.labor_transactions = []
        self.labor_transactions_history = []
        self.capital_transactions = []
        self.capital_transactions_history = []
        self.time_horizon = self.config.PLANNING_HORIZON
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


        # Create Firm1 instances
        for i in range(self.num_firm1):
            firm = Firm1(self.num_workers + self.num_firm2 + i, self)
            self.schedule.add(firm)


    def step(self):
        # Clear previous transactions

        # Execute the step
        self.schedule.step()

        # Update firms
        for agent in self.schedule.agents:
            if isinstance(agent, (Firm1, Firm2)):

                agent.update_firm_state()
 
                agent.update_expectations()

                agent.make_production_decision() 

                agent.adjust_labor()

                agent.nash_improvements()

                # First time for wage
                agent.get_zero_profit_conditions()
                if isinstance(agent, Firm2):
                    agent.adjust_investment_demand()


        self.labor_transactions.clear()
        self.capital_transactions.clear()
        self.consumption_transactions.clear()
        # Execute markets
        self.execute_labor_market()
        # re-calculate zero-profit conditions for capital
        self.execute_capital_market()
        # re-calculate zero-profit conditions for consumption
        self.execute_consumption_market()

        # Adjust production and prices
        for agent in self.schedule.agents:
            if isinstance(agent, (Firm1,Firm2)):
                agent.adjust_production()
                agent.nash_improvements()
                agent.pay_wages()
                agent.calculate_profit()

        # Collect data
        self.data_collector.datacollector.collect(self)
        print(f"Step {self.step_count} completed")
        self.step_count += 1
        if self.config.TIME_HORIZON - self.step_count <= self.config.PLANNING_HORIZON:
          self.time_horizon = self.config.TIME_HORIZON - self.step_count


    def execute_labor_market(self):
            
        buyers = [(firm.labor_demand, firm.desireds['wage'], firm, firm.zero_profit_conditions['wage'], firm.preference_mode
        )
                  for firm in self.schedule.agents
                  if isinstance(firm, (Firm1, Firm2)) and firm.labor_demand > 0]

        sellers = [(worker.available_hours(), worker.desired_wage, worker, worker.get_min_wage(), worker.skills, worker.skillscarbon)
                   for worker in self.schedule.agents
                   if isinstance(worker, Worker) and worker.available_hours() > 0]
        #print("buyers", buyers, "sellers", sellers)
        #("Labor Market Transaction", transactions)
        buyer_demand = sum(b[0] for b in buyers) if buyers else 0
        seller_inventory = sum(s[0] for s in sellers) if sellers else 0
        self.labor_supply = seller_inventory
        avg_buyer_price = sum(b[1] for b in buyers) / len(buyers) if buyers else 0
        avg_seller_price = sum(s[1] for s in sellers) / len(sellers) if sellers else 0
        avg_buyer_max = sum(b[3] for b in buyers)/ len(buyers) if buyers else 0
        avg_seller_min = sum(s[3] for s in sellers)/ len(sellers) if sellers else 0
        std_buyer_price = np.std([b[1] for b in buyers]) if buyers else 0
        std_seller_price = np.std([s[1] for s in sellers]) if sellers else 0
        std_buyer_max = np.std([b[3] for b in buyers]) if buyers else 0
        std_seller_min = np.std([s[3] for s in sellers]) if sellers else 0

        self.pre_labor_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price,avg_buyer_max, avg_seller_min, std_buyer_price, std_seller_price, std_buyer_max, std_seller_min])


        print(self.pre_labor_transactions)
        transactions = market_matching(buyers, sellers)
        self.labor_transactions = transactions
        labor_transactions_history = np.array([len(transactions), sum(t[2] for t in transactions), sum(t[3] for t in transactions)])
        self.labor_transactions_history.append(labor_transactions_history)
        for firm, worker, hours, price, a_round, market_advantage in transactions:
            # Round hours to the nearest integer
            hours = round(hours)
            if hours > 0:
                firm.hire_worker(worker, price, hours, a_round, market_advantage)

        # Update labor demand for firms
        for firm in self.schedule.agents:
            if isinstance(firm, (Firm1, Firm2)):
                firm.labor_demand = max(0, firm.labor_demand - sum(t[2] for t in transactions if t[0] == firm))

    def execute_capital_market(self):
        
        buyers = [(firm.investment_demand, firm.desireds['capital_price'], firm, firm.zero_profit_conditions['capital_price'], firm.preference_mode)
                  for firm in self.schedule.agents
                  if isinstance(firm, Firm2) and firm.investment_demand > 0]


        sellers = []
        for firm in self.schedule.agents:
            match firm:
                case Firm1() if firm.inventory > 0:
                    sellers.append((firm.inventory, firm.desireds['price'], firm, firm.zero_profit_conditions['price'], firm.productivity, firm.carbon_intensity))
                case Firm2() if firm.capital_inventory > 0:
                    sellers.append((firm.capital_inventory,firm.capital_resale_price, firm, 0.1, firm.productivity, firm.carbon_intensity))
        buyer_demand = sum(b[0] for b in buyers)  if buyers else 0
        seller_inventory = sum(s[0] for s in sellers)  if sellers else 0
        self.capital_supply = seller_inventory
        avg_buyer_price = sum(b[1] for b in buyers) / len(buyers) if buyers else 0
        avg_seller_price = sum(s[1] for s in sellers) / len(sellers) if sellers else 0
        avg_buyer_max = sum(b[3] for b in buyers)/ len(buyers) if buyers else 0
        avg_seller_min = sum(s[3] for s in sellers)/ len(sellers) if sellers else 0
        std_buyer_price = np.std([b[1] for b in buyers]) if buyers else 0
        std_seller_price = np.std([s[1] for s in sellers]) if sellers else 0
        std_buyer_max = np.std([b[3] for b in buyers]) if buyers else 0
        std_seller_min = np.std([s[3] for s in sellers]) if sellers else 0
        self.pre_capital_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price, avg_buyer_max, avg_seller_min, std_buyer_price, std_seller_price, std_buyer_max, std_seller_min])


        transactions = market_matching(buyers, sellers)
        self.capital_transactions = transactions
        captial_transactions_history = np.array([len(transactions), sum(t[2] for t in transactions), sum(t[3] for t in transactions)])
        self.capital_transactions_history.append(captial_transactions_history)
        for buyer, seller, quantity, price, a_round, market_advantage in transactions:
            buyer.buy_capital(quantity, price, a_round, market_advantage)
            seller.sell_goods(quantity, price, a_round, market_advantage)


    def execute_consumption_market(self):
        #print("Executing consumption market")
   
        buyers = [(worker.desired_consumption, worker.desired_price, worker, worker.get_max_consumption_price(), worker.preference_mode)
                  for worker in self.schedule.agents
                  if isinstance(worker, Worker) and worker.savings > 0]

        sellers = [(min(firm.inventory, firm.optimals['sales']), firm.desireds['price'], firm, firm.zero_profit_conditions['price'], firm.quality, firm.carbon_intensity)
                   for firm in self.schedule.agents

                   if isinstance(firm, Firm2) and firm.inventory > 0]

        if sellers:
            min_price = max(sellers, key=lambda x: x[1])[1]
            max_inventory = max(sellers, key=lambda x: x[0])[0]
            #print("Min price, max inventory", min_price, max_inventory)
        #print("Sellers", sellers)


        buyer_demand = sum(b[0] for b in buyers) if buyers else 0
        seller_inventory = sum(s[0] for s in sellers) if sellers else 0
        self.consumption_supply = seller_inventory
        avg_buyer_price = sum(b[1] for b in buyers) / len(buyers) if buyers else 0
        avg_seller_price = sum(s[1] for s in sellers) / len(sellers) if sellers else 0
        avg_buyer_max = sum(b[3] for b in buyers)/ len(buyers) if buyers else 0
        avg_seller_min = sum(s[3] for s in sellers)/ len(sellers) if sellers else 0
        std_buyer_price = np.std([b[1] for b in buyers]) if buyers else 0
        std_seller_price = np.std([s[1] for s in sellers]) if sellers else 0    
        std_buyer_max = np.std([b[3] for b in buyers]) if buyers else 0
        std_seller_min = np.std([s[3] for s in sellers]) if sellers else 0

        self.pre_consumption_transactions = np.array([buyer_demand, seller_inventory, avg_buyer_price, avg_seller_price, avg_buyer_max, avg_seller_min, std_buyer_price, std_seller_price, std_buyer_max, std_seller_min])

        transactions = market_matching(buyers, sellers)

        self.consumption_transactions = transactions
        #t[0] is buyer, t[1] is seller, t[2] is quantity, t[3] is price
        consumption_transactions_history = np.array([len(transactions), sum(t[2] for t in transactions), sum(t[3] for t in transactions)])
        self.consumption_transactions_history.append(consumption_transactions_history)

        if consumption_transactions_history[1] < seller_inventory:
          print(f"Unsold Stuff{(seller_inventory - consumption_transactions_history[1])}")
        else:
          print(f"Seller Inventory: {seller_inventory} Sold: {consumption_transactions_history[1]}")
        for buyer, seller, quantity, price, a_round, market_advantage in transactions:
            buyer.consume(quantity, price, a_round, market_advantage)
            seller.sell_goods(quantity, price, a_round, market_advantage)

    def get_total_labor_supply(self):
        return self.labor_supply    

    def  get_total_capital_supply(self):
        return self.capital_supply

    def get_total_consumption_supply(self):
        return self.consumption_supply
