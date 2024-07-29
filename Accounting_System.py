from mesa import Agent
import csv
from datetime import datetime

class AccountingSystem:
    def __init__(self):
        self.assets = {}
        self.liabilities = {}
        self.equity = {}
        self.income = {}
        self.expenses = {}

    def record_transaction(self, from_account, to_account, amount):
        self.assets[from_account] = self.assets.get(from_account, 0) - amount
        self.assets[to_account] = self.assets.get(to_account, 0) + amount

    def update_balance_sheet(self):
        self.equity['retained_earnings'] = sum(self.assets.values()) - sum(self.liabilities.values())

    def record_income(self, account, amount):
        self.income[account] = self.income.get(account, 0) + amount

    def record_expense(self, account, amount):
        self.expenses[account] = self.expenses.get(account, 0) + amount

    def calculate_profit(self):
        return sum(self.income.values()) - sum(self.expenses.values())
    
    def get_total_demand(self):
        return self.assets.get('accounts_receivable', 0) + self.income.get('sales', 0)

    def get_total_production(self):
        return self.assets.get('inventory', 0) + self.income.get('sales', 0)

class GlobalAccountingSystem:
    def __init__(self):
        self.total_labor = 0
        self.total_capital = 0
        self.total_goods = 0
        self.total_money = 0
        self.market_demand = []
        self.capital_prices = []
        self.wages = []
        self.firms = []
        self.consumption_good_prices = []

    def record_labor_transaction(self, firm, worker, quantity, price):
        self.total_money += price  # Money flows from firm to worker
        self.wages.append(price)

    def record_capital_transaction(self, buyer, seller, quantity, price):
        self.total_capital += quantity  # Capital flows from seller to buyer
        self.capital_prices.append(price)

    def record_consumption_transaction(self, buyer, seller, quantity, price):
        self.total_goods -= quantity  # Goods flow from seller to buyer
        self.consumption_good_prices.append(price)

    def record_market_demand(self, demand):
        self.market_demand.append(demand)

    def get_total_labor(self):
        return self.total_labor

    def get_total_capital(self):
        return self.total_capital

    def get_total_goods(self):
        return self.total_goods

    def get_total_money(self):
        return self.total_money

    def get_average_market_demand(self):
        return sum(self.market_demand) / len(self.market_demand) if self.market_demand else 0

    def get_average_capital_price(self):
        return sum(self.capital_prices) / len(self.capital_prices) if self.capital_prices else 0

    def get_average_wage(self):
        return sum(self.wages) / len(self.wages) if self.wages else 0

    def get_average_consumption_good_price(self):
        return sum(self.consumption_good_prices) / len(self.consumption_good_prices) if self.consumption_good_prices else 0
    
    def register_firm(self, firm):
        self.firms.append(firm)

    def get_total_demand(self):
        return sum(firm.accounts.get_total_demand() for firm in self.firms)

    def get_total_production(self):
        return sum(firm.accounts.get_total_production() for firm in self.firms)


    def check_consistency(self):
        # Implement overall consistency checks here
        # For example, ensure that total assets equal total liabilities plus equity
        total_assets = self.total_capital + self.total_goods + self.total_money
        total_liabilities_and_equity = self.calculate_total_liabilities_and_equity()
        assert abs(total_assets - total_liabilities_and_equity) < 1e-6, "Balance sheet inconsistency detected"

    def calculate_total_liabilities_and_equity(self):
        # Implement the calculation of total liabilities and equity
        # This should include all forms of liabilities (e.g., loans) and all forms of equity
        # For now, we'll return the total assets as a placeholder
        return self.total_capital + self.total_goods + self.total_money

    def reset_period_data(self):
        # Reset data that should be cleared after each period
        self.market_demand.clear()
        self.capital_prices.clear()
        self.wages.clear()
        self.consumption_good_prices.clear()

    def dump_data_to_csv(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"global_accounting_data_{timestamp}.csv"

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Variable', 'Value'])  # Header row
            
            # Write all the relevant data
            writer.writerow(['Total Labor', self.total_labor])
            writer.writerow(['Total Capital', self.total_capital])
            writer.writerow(['Total Goods', self.total_goods])
            writer.writerow(['Total Money', self.total_money])
            writer.writerow(['Average Market Demand', self.get_average_market_demand()])
            writer.writerow(['Average Capital Price', self.get_average_capital_price()])
            writer.writerow(['Average Wage', self.get_average_wage()])
            writer.writerow(['Average Consumption Good Price', self.get_average_consumption_good_price()])
            writer.writerow(['Total Demand', self.get_total_demand()])
            writer.writerow(['Total Production', self.get_total_production()])

            # Write time series data
            writer.writerow([])  # Empty row for separation
            writer.writerow(['Time Series Data'])
            writer.writerow(['Step', 'Market Demand', 'Capital Price', 'Wage', 'Consumption Good Price'])
            for step in range(max(len(self.market_demand), len(self.capital_prices), len(self.wages), len(self.consumption_good_prices))):
                writer.writerow([
                    step,
                    self.market_demand[step] if step < len(self.market_demand) else '',
                    self.capital_prices[step] if step < len(self.capital_prices) else '',
                    self.wages[step] if step < len(self.wages) else '',
                    self.consumption_good_prices[step] if step < len(self.consumption_good_prices) else ''
                ])

        print(f"Data has been dumped to {filename}")