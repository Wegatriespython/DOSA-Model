from Config import config
from Wageoffer import WageOffer

class BaseFirm:
    def __init__(self):
        self.capital = config.INITIAL_CAPITAL
        self.productivity = config.INITIAL_PRODUCTIVITY
        self.price = config.INITIAL_PRICE
        self.inventory = config.INITIAL_INVENTORY
        self.workers = []
        self.demand = config.INITIAL_DEMAND
        self.production = 0
        self.desired_workers = 0
        self.wage_offers = []
        self.inventory_threshold = config.INVENTORY_THRESHOLD
        self.markup_rate = config.MARKUP_RATE
#this is a change 
    def update_labor_demand(self):
        if self.inventory > self.inventory_threshold:
            self.desired_workers = max(1, int(len(self.workers) * 0.95))
        elif self.demand > self.production:
            self.desired_workers = min(int(len(self.workers) * 1.05), len(self.workers) + 1)
        else:
            self.desired_workers = len(self.workers)

        workers_to_change = self.desired_workers - len(self.workers)
        if workers_to_change < 0:
            self.fire_workers(-workers_to_change)

    def fire_workers(self, num_to_fire):
        for _ in range(min(num_to_fire, len(self.workers))):
            worker = self.workers.pop()
            worker.employed = False
            worker.employer = None


    def produce(self):
        self.production = self.capital * self.productivity * len(self.workers) * config.PRODUCTION_FACTOR
        self.inventory += self.production


    def set_prices(self):
        self.price = (1 + self.markup_rate) * self.get_production_cost()

    def get_production_cost(self):
        total_wages = sum([worker.get_wage() for worker in self.workers])
        return total_wages / self.production if self.production > 0 else 0

    def get_wage_offer(self, worker):
        return worker.get_skills() * config.WAGE_OFFER_FACTOR

    def make_wage_offers(self, worker_applications):
        self.wage_offers = []
        for application in worker_applications:
            if application.firm == self:
                wage_offer = self.get_wage_offer(application.worker)
                offer = WageOffer(application.worker, self, wage_offer)
                self.wage_offers.append(offer)
                application.worker.offers.append(offer)  # Add this line
        

    def update_state(self):
        if self.workers:
            total_wages = sum([worker.get_wage() for worker in self.workers])
            self.inventory -= total_wages
        self.update_labor_demand()