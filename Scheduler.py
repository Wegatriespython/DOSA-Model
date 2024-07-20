# Scheduler.py

class Scheduler:
    def __init__(self, economy):
        self.economy = economy
        self.time = 0

    def step(self):
        self.time += 1

        # 1. Firm1 activities
        for firm in self.economy.firm1s:
            firm.innovate()
            firm.produce()
            firm.set_prices()
            firm.update_state(self.economy.firm2s)

        # 2. Firm2 activities
        for firm in self.economy.firm2s:
            firm.plan_production(self.economy.workers) 
            firm.invest()
            firm.produce() 
            firm.set_prices()

        # 3. Labor market
        self.economy.labor_market_matching()

        # 4. Production
        for firm in self.economy.firm1s + self.economy.firm2s:
            firm.produce()

        # 5. Capital goods market
        self.economy.capital_goods_market()

        # 6. Goods market
        self.economy.goods_market_clearing()
        
        # Consumption
        print("DEBUG: Starting consumption phase")
        for worker in self.economy.workers:
            worker.consume(self.economy.firm2s)
        print("DEBUG: Finished consumption phase")
        total_consumption = sum(worker.consumption for worker in self.economy.workers)
        print(f"DEBUG: Total consumption this step: {total_consumption}")
              
        # 7. Update agent states
        for firm in self.economy.firm1s:
            firm.update_state(self.economy.firm2s)
        for firm in self.economy.firm2s:
            firm.update_state()
        for worker in self.economy.workers:
            worker.update_state()

        self.economy.global_productivity = self.economy.calculate_global_productivity()
        self.economy.data_collector.collect_data(self.economy)
        
        for firm in self.economy.firm1s + self.economy.firm2s:
            firm.update_labor_demand()
        # Update economy-wide attributes
        self.update_economy_attributes()

        # Collect data
        self.economy.data_collector.collect_data(self.economy)

        # Print statistics at the end of each step
        self.print_statistics()



    def print_statistics(self):
        data = self.economy.data_collector.data[-1]  # Get the latest data point
        
        print(f"\nTimestep {self.time} Stats:")
        print(f"Total Firm1 production: {sum(firm['production'] for firm in data['firm1_data'])}")
        print(f"Total Firm2 production: {sum(firm['production'] for firm in data['firm2_data'])}")
        print(f"Total employed workers: {sum(1 for worker in data['worker_data'] if worker['employed'])}")
        print(f"Average wage: {sum(worker['wage'] for worker in data['worker_data']) / len(data['worker_data']):.2f}")
        print(f"Total capital: {sum(firm['capital'] for firm in data['firm1_data'] + data['firm2_data'])}")
        print(f"Total inventory: {sum(firm['inventory'] for firm in data['firm1_data'] + data['firm2_data'])}")   

    def update_economy_attributes(self):
        self.economy.total_demand = sum(firm.demand for firm in self.economy.firm1s + self.economy.firm2s)
        self.economy.total_supply = sum(firm.inventory for firm in self.economy.firm1s + self.economy.firm2s)
        self.economy.global_productivity = self.calculate_global_productivity()

    def calculate_global_productivity(self):
        total_output = sum(firm.production for firm in self.economy.firm1s + self.economy.firm2s)
        total_labor = sum(len(firm.workers) for firm in self.economy.firm1s + self.economy.firm2s)
        return total_output / total_labor if total_labor > 0 else 1

    def run_simulation(self, num_steps):
        for _ in range(num_steps):
            self.step()
        
        # Write simulation data to CSV after all steps are complete
        self.economy.write_data_to_csv("V:/Python Port/simulation_results.csv")
