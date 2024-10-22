import numpy as np
from Src.Utilities.market_matching import market_matching
from Src.Utilities.adaptive_expectations import autoregressive
from Turbo_Firm import *
from Turbo_Worker import *
from Turbo_utils import *
import matplotlib.pyplot as plt
from best_response import best_response, find_equilibrium

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
    max_iterations = 600
    tolerance = 1e-3

    # Create workers and firms using their initialized values
    workers = [Worker(i, initial_wealth=2) for i in range(num_workers)]
    firms = [Firm(i + num_workers, initial_wealth=5) for i in range(num_firms)]

    # Initialize lists to store data for plotting
    labor_market_data = []
    consumption_market_data = []

    # Initial prices guess
    initial_prices = np.array([1, 1])  # [wage, goods_price]

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")

        # Prepare worker and firm parameters for optimization
        worker_params = [worker.get_utility_params() for worker in workers]
        firm_params = [firm.get_profit_params() for firm in firms]

        # Find equilibrium prices
        try:
            equilibrium_prices = find_equilibrium(worker_params, firm_params, initial_prices)
            new_wage, new_goods_price = equilibrium_prices
        except ValueError as e:
            print(f"Error in finding equilibrium: {e}")
            new_wage, new_goods_price = initial_prices

        # Update worker and firm decisions based on equilibrium prices
        for worker, params in zip(workers, worker_params):
            params['wage'] = new_wage
            params['price'] = new_goods_price
            try:
                optimal_decision, _ = best_response('worker', params)
                worker.update_decision(optimal_decision)
            except ValueError as e:
                print(f"Error in worker optimization: {e}")

        for firm, params in zip(firms, firm_params):
            params['wage'] = new_wage
            params['price'] = new_goods_price
            try:
                optimal_decision, _ = best_response('firm', params)
                firm.update_decision(optimal_decision)
            except ValueError as e:
                print(f"Error in firm optimization: {e}")

        # Update market statistics
        labor_market_stats = {
            'demand': sum(firm.labor_demand for firm in firms),
            'supply': sum(worker.working_hours for worker in workers),
            'price': new_wage,
            'bid': np.mean([firm.labor_bid for firm in firms]),
            'ask': np.mean([worker.desired_wage for worker in workers]),
            'bid_max': np.mean([firm.get_max_labor_price() for firm in firms]),
            'ask_min': np.mean([worker.get_min_wage() for worker in workers])
        }

        consumption_market_stats = {
            'demand': sum(worker.desired_consumption for worker in workers),
            'supply': sum(firm.production for firm in firms),
            'price': new_goods_price,
            'bid': np.mean([worker.desired_price for worker in workers]),
            'ask': np.mean([firm.consumption_ask for firm in firms]),
            'bid_max': np.mean([worker.get_max_consumption_price() for worker in workers]),
            'ask_min': np.mean([firm.get_min_consumption_price() for firm in firms])
        }

        # Update expectations
        for worker in workers:
            worker.update_expectations(labor_market_stats, consumption_market_stats)

        for firm in firms:
            firm.update_expectations(labor_market_stats, consumption_market_stats)

        # Check for convergence
        price_change = np.linalg.norm(equilibrium_prices - initial_prices) / np.linalg.norm(initial_prices)
        print(f"Price change: {price_change:.4f}")

        if price_change < tolerance:
            print(f"Convergence achieved in {iteration + 1} iterations.")
            break

        if iteration == max_iterations - 1:
            print(f"Max iterations ({max_iterations}) reached.")

        # Update initial prices for next iteration
        initial_prices = equilibrium_prices

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
