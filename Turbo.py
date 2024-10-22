import numpy as np
from Src.Utilities.market_matching import market_matching
from Src.Utilities.Turbo_profit import profit_maximization
from Src.Utilities.Turbo_utility import maximize_utility
from Src.Utilities.adaptive_expectations import autoregressive
from Turbo_Firm import *
from Turbo_Worker import *
from Turbo_utils import *
import matplotlib.pyplot as plt


def info_dump(agent):
    print(f"{'=' * 40}")
    if isinstance(agent, Worker):
        print(f"Info dump for Worker {agent.unique_id}")
    elif isinstance(agent, Firm):
        print(f"Info dump for Firm {agent.id}")
    else:
        print(f"Info dump for Unknown Agent Type")
    print(f"{'=' * 40}")
    
    for attr, value in agent.__dict__.items():
        if attr.startswith('__'):
            continue
        
        if isinstance(value, dict):
            print(f"{attr}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list) and len(value) > 10:
            print(f"{attr}: [list with {len(value)} items]")
        else:
            print(f"{attr}: {value}")
    
    print(f"{'=' * 40}")

def simulate_market():
    num_workers = 30
    num_firms = 5
    max_iterations = 100
    tolerance = 1e-3

    # Create workers and firms using their initialized values
    workers = [Worker(i, initial_wealth=2) for i in range(num_workers)]
    firms = [Firm(i + num_workers, initial_wealth=5) for i in range(num_firms)]

    # Initialize lists to store data for plotting
    labor_market_data = []
    consumption_market_data = []

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")

        # Update expectations and decisions
        for worker in workers:
            try:
                utility_params = worker.get_utility_params()
                results = maximize_utility(utility_params)


                worker.desired_consumption, worker.working_hours, _, _ = [arr[0] for arr in results]
            except ValueError as e:
                print(f"Error in worker utility maximization: {e}")
                worker.desired_consumption, worker.working_hours = 1, 8  # Default values

        for firm in firms:
            try:
                print(f"calling profit maximization with {firm.get_profit_params()}")
                profit_params = firm.get_profit_params()
                result = profit_maximization(profit_params)
                print(f"result: {result}")

                if result:
                    firm.labor_demand = result['optimal_labor'] * 16
                    firm.production = result['optimal_production']
                else:
                    print("Profit maximization returned None")
                    firm.labor_demand, firm.production = 1, 1  # Default values
            except ValueError as e:
                print(f"Error in firm profit maximization: {e}")
                firm.labor_demand, firm.production = 1, 1  # Default values


        labor_price_params = {
            'demand': sum(firm.labor_demand for firm in firms), 
            'supply': sum(worker.working_hours for worker in workers),
            'bid': np.mean([firm.labor_bid for firm in firms]),
            'ask': np.mean([worker.desired_wage for worker in workers]),
            'bid_max': np.mean([firm.get_max_labor_price() for firm in firms]),
            'ask_min': np.mean([worker.get_min_wage() for worker in workers])
        }
        # Labor market
        labor_market_stats = {
            'demand': labor_price_params['demand'],
            'supply': labor_price_params['supply'],
            'price': find_equilibrium_price(**labor_price_params),
            'bid': labor_price_params['bid'],
            'ask': labor_price_params['ask'],
            'bid_max': labor_price_params['bid_max'],
            'ask_min': labor_price_params['ask_min'],
            'round_num': 1,
            'advantage': '',
            'market_type': 'labor'
        }


        # Consumption market
        Consumption_price_params = {
            'demand': sum(worker.desired_consumption for worker in workers),
            'supply': sum(firm.production for firm in firms),
            'bid': np.mean([worker.desired_price for worker in workers]),
            'ask': np.mean([firm.consumption_ask for firm in firms]),
            'bid_max': np.mean([worker.get_max_consumption_price() for worker in workers]),
            'ask_min': np.mean([firm.get_min_consumption_price() for firm in firms])
        }
        consumption_market_stats = {
            'demand': Consumption_price_params['demand'],
            'supply': Consumption_price_params['supply'],
            'price': find_equilibrium_price(**Consumption_price_params),
            'bid': Consumption_price_params['bid'],
            'ask': Consumption_price_params['ask'],
            'bid_max': Consumption_price_params['bid_max'],
            'ask_min': Consumption_price_params['ask_min'],
            'round_num': 1,
            'advantage': '',
            'market_type': 'consumption'
        }

        # Labor market strategic adjustments
        for worker in workers:
            price_decision_data = {**labor_market_stats, 'pvt_res_price': worker.get_min_wage(), 'is_buyer': False, 'previous_price': worker.desired_wage}
            worker.desired_wage = best_response_exact(price_decision_data)

        for firm in firms:
            price_decision_data = {**labor_market_stats, 'pvt_res_price': firm.get_max_labor_price(), 'is_buyer': True, 'previous_price': firm.labor_bid}
            firm.labor_bid = best_response_exact(price_decision_data)

        # Consumption market strategic adjustments
        for worker in workers:
            price_decision_data = {**consumption_market_stats, 'pvt_res_price': worker.get_max_consumption_price(), 'is_buyer': True, 'previous_price': worker.desired_price}
            worker.desired_price = best_response_exact(price_decision_data)

        for firm in firms:
            price_decision_data = {**consumption_market_stats, 'pvt_res_price': firm.get_min_consumption_price(), 'is_buyer': False, 'previous_price': firm.consumption_ask}
            firm.consumption_ask = best_response_exact(price_decision_data)

        # Market matching
        labor_transactions = market_matching(
            [(firm.labor_demand, firm.labor_bid, firm.id, firm.get_max_labor_price(), 0) for firm in firms],
            [(worker.working_hours, worker.desired_wage, worker.unique_id, worker.get_min_wage(), 0, 0) for worker in workers]
        )

        consumption_transactions = market_matching(
            [(worker.desired_consumption, worker.desired_price, worker.unique_id, worker.get_max_consumption_price(), 0) for worker in workers],
            [(firm.production, firm.consumption_ask, firm.id, firm.get_min_consumption_price(), 0, 0) for firm in firms]
        )

        # Update expectations
        new_labor_price = np.mean([t[3] for t in labor_transactions]) if labor_transactions else labor_market_stats['price']
        new_consumption_price = np.mean([t[3] for t in consumption_transactions]) if consumption_transactions else consumption_market_stats['price']

        # Update the existing dictionaries instead of creating new ones
        labor_market_stats.update({
            'price': new_labor_price,
            'quantity': sum(t[2] for t in labor_transactions) if labor_transactions else 0

        })

        consumption_market_stats.update({
            'price': new_consumption_price,
            'quantity': sum(t[2] for t in consumption_transactions) if consumption_transactions else 0

        })

        for worker in workers:
            worker.update_expectations(labor_market_stats, consumption_market_stats)

        for firm in firms:
            firm.update_expectations(labor_market_stats, consumption_market_stats)

        # Check for convergence
        labor_price_change = abs(new_labor_price - labor_market_stats['price']) / labor_market_stats['price']
        consumption_price_change = abs(new_consumption_price - consumption_market_stats['price']) / consumption_market_stats['price']

        print(f"Labor price change: {labor_price_change:.4f}")
        print(f"Consumption price change: {consumption_price_change:.4f}")

        if labor_price_change < tolerance and consumption_price_change < tolerance:
            print(f"Convergence achieved in {iteration + 1} iterations.")

        if iteration == max_iterations - 1:
            print(f"Max iterations ({max_iterations}) reached.")

        # After updating expectations and decisions
        if iteration % 10 == 0:  # Print info dump every 10 iterations
            print("\nWorker Info Dump:")
            info_dump(workers[0])  # Dump info for the first worker
            print("\nFirm Info Dump:")
            info_dump(firms[0])  # Dump info for the first firm

        # Store data for plotting
        labor_market_data.append({
            'iteration': iteration,
            'demand': labor_market_stats['demand'],
            'supply': labor_market_stats['supply'],
            'price': labor_market_stats['price'],
            'bid': labor_market_stats['bid'],
            'ask': labor_market_stats['ask'],
            'bid_max': labor_market_stats['bid_max'],
            'ask_min': labor_market_stats['ask_min']
        })

        consumption_market_data.append({
            'iteration': iteration,
            'demand': consumption_market_stats['demand'],
            'supply': consumption_market_stats['supply'],
            'price': consumption_market_stats['price'],
            'bid': consumption_market_stats['bid'],
            'ask': consumption_market_stats['ask'],
            'bid_max': consumption_market_stats['bid_max'],
            'ask_min': consumption_market_stats['ask_min']
        })

    return workers, firms, labor_market_data, consumption_market_data

# Run the simulation
workers, firms, labor_market_data, consumption_market_data = simulate_market()

# Print final expectations
print("\nFinal Labor Price Expectations:")
print(f"Workers: {np.mean([w.worker_expectations['price']['labor'][0] for w in workers]):.2f}")
print(f"Firms: {np.mean([f.labor_price_expectations[0] for f in firms]):.2f}")

print("\nFinal Consumption Price Expectations:")
print(f"Workers: {np.mean([w.worker_expectations['price']['consumption'][0] for w in workers]):.2f}")
print(f"Firms: {np.mean([f.consumption_price_expectations[0] for f in firms]):.2f}")

# Create plots
def create_market_plots(market_data, market_name):
    iterations = [d['iteration'] for d in market_data]
    
    # Supply vs Demand plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, [d['demand'] for d in market_data], label='Demand')
    plt.plot(iterations, [d['supply'] for d in market_data], label='Supply')
    plt.xlabel('Iteration')
    plt.ylabel('Quantity')
    plt.title(f'{market_name} Market: Supply vs Demand')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{market_name.lower()}_supply_demand.png')
    plt.close()

    # Prices plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, [d['price'] for d in market_data], label='Equilibrium Price')
    plt.plot(iterations, [d['bid'] for d in market_data], label='Bid')
    plt.plot(iterations, [d['ask'] for d in market_data], label='Ask')
    plt.plot(iterations, [d['bid_max'] for d in market_data], label='Max Bid')
    plt.plot(iterations, [d['ask_min'] for d in market_data], label='Min Ask')
    plt.xlabel('Iteration')
    plt.ylabel('Price')
    plt.title(f'{market_name} Market: Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{market_name.lower()}_prices.png')
    plt.close()

# Create plots for both markets
create_market_plots(labor_market_data, 'Labor')
create_market_plots(consumption_market_data, 'Consumption')

print("\nPlots have been saved as PNG files in the current directory.")
