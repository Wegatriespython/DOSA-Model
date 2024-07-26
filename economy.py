from Config import config
import random
from Worker import Worker
from firm1 import Firm1
from firm2 import Firm2
from Stats import DataCollector

class Economy:
    def __init__(self, num_workers, num_firm1, num_firm2):
        self.workers = [Worker() for _ in range(num_workers)]
        self.firm1s = [Firm1() for _ in range(num_firm1)]
        self.firm2s = [Firm2() for _ in range(num_firm2)]
        self.time = 0
        self.total_demand = 0 
        self.total_supply = 0
        self.global_productivity = 1
        self.data_collector = DataCollector()

    def labor_market_matching(self):
        all_firms = self.firm1s + self.firm2s

        for firm in all_firms:
            firm.calculate_budget()
            workers_to_fire = []

            for worker in firm.workers:
                if firm.budget >= worker.wage:
                    firm.budget -= worker.wage
                    worker.savings += worker.wage
                else:
                    workers_to_fire.append(worker)
            
            for worker in workers_to_fire:
                firm.workers.remove(worker)
                worker.employed = False
                worker.employer = None
                worker.wage = 0

        available_workers = [w for w in self.workers if not w.employed]
        random.shuffle(available_workers)

        for worker in available_workers:
            hiring_firms = [f for f in all_firms if f.budget > config.MINIMUM_WAGE]
            if hiring_firms:
                hiring_firm = random.choice(hiring_firms)
                wage = min(hiring_firm.budget, config.INITIAL_WAGE)
                worker.employed = True
                worker.employer = hiring_firm
                worker.wage = wage
                hiring_firm.workers.append(worker)
                hiring_firm.budget -= wage
                hiring_firm.desired_workers -= 1

    def capital_goods_market(self):
        for firm2 in self.firm2s:
            if firm2.investment_demand > 0:
                for firm1 in self.firm1s:
                    if firm1.inventory > 0:
                        quantity = min(firm2.investment_demand, firm1.inventory)
                        capital_sold = firm1.fulfill_order(quantity)
                        firm2.capital += capital_sold
                        firm2.investment_demand -= capital_sold
                        payment = capital_sold * firm1.price
                        firm2.sales -= payment
                        firm1.sales += payment
                    if firm2.investment_demand == 0:
                        break

    def goods_market_clearing(self):
        for firm in self.firm1s:
            quantity_change = min(firm.demand, firm.inventory)
            firm.inventory -= quantity_change
            firm.quantity_sold += quantity_change

    def consumption_market_clearing(self):
        total_desired_consumption = 0
        worker_consumptions = []

        for worker in self.workers:
            worker.satiated = False
            desired_consumption = worker.consume(self.firm2s)
            total_desired_consumption += desired_consumption
            worker_consumptions.append((worker, desired_consumption))

        total_inventory = sum(firm.inventory for firm in self.firm2s)
        
        if total_inventory > 0:
            print("Entering consumption market clearing")
            for firm in self.firm2s:
                firm_share = firm.inventory / total_inventory
                print("Total desired consumption: ", total_desired_consumption)
                print("Firm share: ", firm_share)
                firm_consumption = total_desired_consumption * firm_share
                actual_consumption = min(firm_consumption, firm.inventory)
                firm.inventory -= actual_consumption
                firm.quantity_sold += actual_consumption
                print("firm price: ", firm.price)
                firm.sales += actual_consumption * firm.price
                if firm.sales > 0: 
                    print("SOLD firm made sale")

                # Distribute this firm's consumption among workers
                for worker, desired in worker_consumptions:
                    if actual_consumption > 0 and desired > 0:
                        worker_consumption = min(desired, actual_consumption)
                        worker.update_savings_and_consumption(worker_consumption, firm.price)
                        actual_consumption -= worker_consumption
                        desired -= worker_consumption
                        if desired == 0:
                            worker.satiated = True

        # Check if any workers are still not satiated
        for worker, remaining_desired in worker_consumptions:
            if remaining_desired > 0:
                worker.satiated = False


  


    def update_global_state(self):
        self.total_demand = sum(firm.demand for firm in self.firm1s + self.firm2s)
        self.total_supply = sum(firm.inventory for firm in self.firm1s + self.firm2s)
        self.global_productivity = self.calculate_global_productivity()

    def calculate_global_productivity(self):
        total_output = sum([firm.production for firm in self.firm1s + self.firm2s])
        total_labor = len([worker for worker in self.workers if worker.employed])
        return total_output / total_labor if total_labor > 0 else 1

    def get_data(self):
        return self.data_collector.get_data()

    def write_data_to_csv(self, filename="simulation_results.csv"):
        self.data_collector.write_to_csv(filename)