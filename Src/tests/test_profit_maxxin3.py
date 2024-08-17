import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from Utilities.Simple_profit_maxxing import profit_maximization

def test_profit_maximization_sensitivity():
    # Test cases based on the specified ranges
    budgets = [2, 10, 18, 25]
    current_labors = [0, 1, 2, 3]
    expected_prices = [0.4, 0.56, 0.72, 0.88, 1.04, 1.2]
    wages = [1, 1.5, 2]

    results = []

    for budget in budgets:
        for current_labor in current_labors:
            for expected_price in expected_prices:
                for wage in wages:
                    params = {
                        "current_capital": 10,
                        "current_labor": current_labor,
                        "current_price": expected_price,
                        "current_productivity": 1,
                        "expected_demand": [10] * 6,
                        "expected_price": [expected_price] * 6,
                        "capital_price": 3,
                        "capital_elasticity": 0.3,
                        "current_inventory": 0,
                        "depreciation_rate": 0.001,
                        "expected_periods": 6,
                        "discount_rate": 0.05,
                        "budget": budget,
                        "wage": wage
                    }

                    result = profit_maximization(**params)

                    if result is not None:
                        results.append({
                            "budget": budget,
                            "current_labor": current_labor,
                            "expected_price": expected_price,
                            "wage": wage,
                            "optimal_production": result['optimal_production'],
                            "optimal_labor": result['optimal_labor'],
                            "optimal_capital": result['optimal_capital'],
                            "optimal_price": result['optimal_price']
                        })

    df = pd.DataFrame(results)
    save_path = 'profit_maximization_sensitivity.csv'
    df.to_csv(save_path, index=False)

    # Plot 1: Impact of Budget and Expected Price on Optimal Production
    plt.figure(figsize=(12, 8))
    for budget in budgets:
        subset = df[df['budget'] == budget]
        plt.plot(subset['expected_price'], subset['optimal_production'], label=f'Budget={budget}')
    plt.xlabel('Expected Price')
    plt.ylabel('Optimal Production')
    plt.title('Impact of Budget and Expected Price on Optimal Production')
    plt.legend()
    plt.savefig('budget_price_production.png')
    plt.close()

    # Plot 2: Impact of Wage and Current Labor on Optimal Production
    plt.figure(figsize=(12, 8))
    for wage in wages:
        subset = df[df['wage'] == wage]
        plt.plot(subset['current_labor'], subset['optimal_production'], label=f'Wage={wage}')
    plt.xlabel('Current Labor')
    plt.ylabel('Optimal Production')
    plt.title('Impact of Wage and Current Labor on Optimal Production')
    plt.legend()
    plt.savefig('wage_labor_production.png')
    plt.close()

    # Plot 3: Heatmap of Optimal Production
    pivot = df.pivot_table(values='optimal_production', index='budget', columns='expected_price', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot, cmap='viridis', aspect='auto')
    plt.colorbar(label='Optimal Production')
    plt.xlabel('Expected Price')
    plt.ylabel('Budget')
    plt.title('Heatmap of Optimal Production')
    plt.xticks(range(len(expected_prices)), expected_prices)
    plt.yticks(range(len(budgets)), budgets)
    plt.savefig('production_heatmap.png')
    plt.close()

    # Analysis of threshold crossing
    production_over_one = df[df['optimal_production'] > 1]

    print("Summary Statistics:")
    print(df.describe())

    print("\nScenarios where production exceeded 1:")
    print(production_over_one[['budget', 'current_labor', 'expected_price', 'wage', 'optimal_production']])

    print("\nCorrelation Matrix:")
    print(df.corr())

if __name__ == "__main__":
    test_profit_maximization_sensitivity()
