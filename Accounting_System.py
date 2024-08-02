class GlobalAccountingSystem:
    def __init__(self):
        self.total_labor = 0
        self.total_capital = 0
        self.total_goods = 0
        self.total_money = 0
        self.market_demand = []
        self.capital_prices = []
        self.wages = []
        self.consumption_good_prices = []
        self.firms = []

    def register_firm(self, firm):
        self.firms.append(firm)

    def get_total_demand(self):
        return sum(firm.expected_demand for firm in self.firms)

    def get_total_production(self):
        return sum(firm.production for firm in self.firms)

    def get_average_market_demand(self):
        return sum(self.market_demand) / len(self.market_demand) if self.market_demand else 0

    def get_average_capital_price(self):
        return sum(self.capital_prices) / len(self.capital_prices) if self.capital_prices else 0

    def get_average_wage(self):
        return sum(self.wages) / len(self.wages) if self.wages else 0

    def get_average_consumption_good_price(self):
        return sum(self.consumption_good_prices) / len(self.consumption_good_prices) if self.consumption_good_prices else 0

    def record_labor_transaction(self, firm, worker, quantity, price):
        self.total_labor += quantity
        self.wages.append(price)

    def record_capital_transaction(self, buyer, seller, quantity, price):
        self.total_capital += quantity
        self.capital_prices.append(price)

    def record_consumption_transaction(self, buyer, seller, quantity, price):
        self.total_goods += quantity
        self.consumption_good_prices.append(price)

    def update_market_demand(self, demand):
        self.market_demand.append(demand)

    def reset_period_data(self):
        self.market_demand = []
        self.capital_prices = []
        self.wages = []
        self.consumption_good_prices = []

    def check_consistency(self):
        # Implement consistency checks if needed
        pass