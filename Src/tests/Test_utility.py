import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utilities.utility_function import maximize_utility
import matplotlib.pyplot as plt

def test_utility_function():
    # Define ranges for each parameter
    savings_range = np.linspace(5, 30, 2)
    wage_range = np.linspace(0.04, 2, 2)
    current_price_range = np.linspace(0.8, 2, 2)
    historical_price_range = np.linspace(1, 1, 1)
    working_hours_range = np.linspace(0, 16, 3)

    # Arrays to store results
    results = []

    # Test the function over the ranges
    for savings in savings_range:
        for wage in wage_range:
            for current_price in current_price_range:
                for historical_price in historical_price_range:
                    for working_hours in working_hours_range:
                        try:
                            result = maximize_utility(
                                savings, wage, current_price, historical_price, working_hours
                            )
                            if result is not None:
                                optimal_consumption, optimal_working_hours, optimal_savings,optimal_future_working_hours  = result
                                results.append({
                                    'savings': savings,
                                    'wage': wage,
                                    'current_price': current_price,
                                    'historical_price': historical_price,
                                    'initial_working_hours': working_hours,
                                    'optimal_consumption': optimal_consumption,
                                    'optimal_working_hours': optimal_working_hours,
                                    'optimal_savings': optimal_savings
                                })
                            else:
                                print(f"Optimization failed for: savings={savings}, wage={wage}, "
                                      f"current_price={current_price}, historical_price={historical_price}, "
                                      f"working_hours={working_hours}")
                        except Exception as e:
                            print(f"Error occurred for: savings={savings}, wage={wage}, "
                                  f"current_price={current_price}, historical_price={historical_price}, "
                                  f"working_hours={working_hours}")
                            print(f"Error message: {str(e)}")

    return results

def plot_results(results):
    # Convert results to numpy arrays for easier plotting
    savings = np.array([r['savings'] for r in results])
    wages = np.array([r['wage'] for r in results])
    optimal_consumption = np.array([r['optimal_consumption'] for r in results])
    optimal_working_hours = np.array([r['optimal_working_hours'] for r in results])
    optimal_savings = np.array([r['optimal_savings'] for r in results])

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # Plot optimal consumption vs savings and wage
    axs[0, 0].scatter(savings, optimal_consumption, alpha=0.5)
    axs[0, 0].set_xlabel('Savings')
    axs[0, 0].set_ylabel('Optimal Consumption')
    axs[0, 0].set_title('Optimal Consumption vs Savings')

    axs[0, 1].scatter(wages, optimal_consumption, alpha=0.5)
    axs[0, 1].set_xlabel('Wage')
    axs[0, 1].set_ylabel('Optimal Consumption')
    axs[0, 1].set_title('Optimal Consumption vs Wage')

    # Plot optimal working hours vs savings and wage
    axs[1, 0].scatter(savings, optimal_working_hours, alpha=0.5)
    axs[1, 0].set_xlabel('Savings')
    axs[1, 0].set_ylabel('Optimal Working Hours')
    axs[1, 0].set_title('Optimal Working Hours vs Savings')

    axs[1, 1].scatter(wages, optimal_working_hours, alpha=0.5)
    axs[1, 1].set_xlabel('Wage')
    axs[1, 1].set_ylabel('Optimal Working Hours')
    axs[1, 1].set_title('Optimal Working Hours vs Wage')

    # Plot optimal savings vs initial savings and wage
    axs[2, 0].scatter(savings, optimal_savings, alpha=0.5)
    axs[2, 0].set_xlabel('Initial Savings')
    axs[2, 0].set_ylabel('Optimal Savings')
    axs[2, 0].set_title('Optimal Savings vs Initial Savings')

    axs[2, 1].scatter(wages, optimal_savings, alpha=0.5)
    axs[2, 1].set_xlabel('Wage')
    axs[2, 1].set_ylabel('Optimal Savings')
    axs[2, 1].set_title('Optimal Savings vs Wage')

    plt.tight_layout()
    plt.savefig('results.png')

if __name__ == "__main__":
    results = test_utility_function()

    # Print summary statistics
    print(f"Total number of successful optimizations: {len(results)}")

    if results:
        optimal_consumption = [r['optimal_consumption'] for r in results]
        optimal_working_hours = [r['optimal_working_hours'] for r in results]
        optimal_savings = [r['optimal_savings'] for r in results]

        print(f"Optimal Consumption - Min: {min(optimal_consumption):.2f}, Max: {max(optimal_consumption):.2f}, Mean: {np.mean(optimal_consumption):.2f}")
        print(f"Optimal Working Hours - Min: {min(optimal_working_hours):.2f}, Max: {max(optimal_working_hours):.2f}, Mean: {np.mean(optimal_working_hours):.2f}")
        print(f"Optimal Savings - Min: {min(optimal_savings):.2f}, Max: {max(optimal_savings):.2f}, Mean: {np.mean(optimal_savings):.2f}")

    plot_results(results)
