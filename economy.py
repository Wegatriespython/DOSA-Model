# economy.py

import random
import math
from Worker import Worker
from firm1 import Firm1
from firm2 import Firm2
from Stats import DataCollector
from Scheduler import Scheduler
class Economy:
    def __init__(self, num_workers, num_firm1, num_firm2):
        self.workers = [Worker() for _ in range(num_workers)]
        self.firm1s = [Firm1() for _ in range(num_firm1)]
        self.firm2s = [Firm2() for _ in range(num_firm2)]
        print(f"Initial number of workers: {len(self.workers)}")
        print(f"Initial number of firm1s: {len(self.firm1s)}")
        print(f"Initial number of firm2s: {len(self.firm2s)}")
        self.time = 0
        self.total_demand = 0 
        self.total_supply = 0
        self.global_productivity = 1
        self.wage_offers = [] # List to store wage offers
        self.worker_applications = [] # List to store worker applications
        self.data_collector = DataCollector()  # Initialize the DataCollector

    def run_simulation(self, num_steps):
        for _ in range(num_steps):
            self.time += 1
            self.step()
    def labor_market_matching(self):
        self.worker_applications = []
        for worker in self.workers:
            if not worker.employed:
                for firm in self.firm1s + self.firm2s:
                    wage = firm.get_wage_offer(worker)
                    self.worker_applications.append(WorkerApplication(worker, firm, wage))
        

        
        self.wage_offers = []
        for firm in self.firm1s + self.firm2s:
            firm.make_wage_offers(self.worker_applications)
        

        
        hired_workers = 0
        for worker in self.workers:
            if not worker.employed:
                best_offer = None
                for offer in worker.offers:
                    if not best_offer or offer.wage > best_offer.wage:
                        best_offer = offer
                if best_offer:
                    worker.employed = True
                    worker.employer = best_offer.firm
                    worker.wage = best_offer.wage
                    best_offer.firm.workers.append(worker)
                    worker.offers = [] # Clear offers for next round
                    hired_workers += 1


    def capital_goods_market(self):
        for firm2 in self.firm2s:
            if firm2.investment_demand > 0:
                for firm1 in self.firm1s:
                    if firm1.inventory > 0:
                        quantity = min(firm2.investment_demand, firm1.inventory)
                        capital_sold = firm1.fulfill_order(quantity)
                        firm2.receive_capital(capital_sold)
                        firm2.investment_demand -= capital_sold
                    if firm2.investment_demand == 0:
                            break
   
    def goods_market_clearing(self):
        # Update total demand and supply for debugging
        self.total_demand = sum([firm.demand for firm in self.firm2s])
        self.total_supply = sum([firm.inventory for firm in self.firm1s])

    def calculate_global_productivity(self):
        total_output = sum([firm.production for firm in self.firm1s + self.firm2s])
        total_labor = len([worker for worker in self.workers if worker.employed])
        if total_labor > 0:
            return total_output / total_labor
        else:
            return 1 
    def get_data(self):
        """
        Returns the collected data.
        """
        return self.data_collector.get_data()
    def write_data_to_csv(self, filename="V:/Python Port/simulation_results.csv"):
        self.data_collector.write_to_csv(filename)
class WorkerApplication:
    def __init__(self, worker, firm, wage):
        self.worker = worker
        self.firm = firm
        self.wage = wage
