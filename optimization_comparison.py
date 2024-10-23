import numpy as np
from Labor_production_analysis import analyze_production_labor
from Profit_function import evaluate_profit, production_function

def compare_optimizations():
    # Common parameters matching Profit_function.py
    params = {
        'capital': 6,
        'price': 1,
        'wage': 0.0625,
        'capital_price': 0,  # Not used in Profit_function
        'depreciation_rate': 0,  # Not used in Profit_function
        'productivity': 1,
        'capital_elasticity': 0.5,  # = 1 - alpha from Profit_function (alpha=0.9)
        'labor_supply': 30,  # Matching max_labor from Profit_function
        'demand': float('inf')  # No demand constraint in Profit_function
    }
    
    # Run Labor_production_analysis optimization
    results_lpa = analyze_production_labor(params)
    
    # Calculate equivalent values using Profit_function approach
    optimal_production_pf = production_function(
        results_lpa['optimal_solution'].labor,
        params['capital'],
        1 - params['capital_elasticity']  # Convert to alpha
    )
    
    profit_pf = evaluate_profit(
        [optimal_production_pf],
        [results_lpa['optimal_solution'].labor],
        params['capital'],
        1 - params['capital_elasticity'],  # Convert to alpha
        params['price'],
        params['wage'],
        0.05,  # discount_rate from Profit_function
        1      # periods from Profit_function
    )
    
    print("=== Comparison Results ===")
    print("\nLabor_production_analysis results:")
    print(f"Optimal Labor: {results_lpa['optimal_solution'].labor:.4f}")
    print(f"Optimal Production: {results_lpa['optimal_solution'].production:.4f}")
    print(f"Optimal Profit: {results_lpa['optimal_solution'].profit:.4f}")
    
    print("\nProfit_function results:")
    print(f"Production at LPA labor: {optimal_production_pf:.4f}")
    print(f"Profit at LPA labor: {profit_pf:.4f}")
    
    print("\nDifferences:")
    print(f"Production Difference: {abs(results_lpa['optimal_solution'].production - optimal_production_pf):.4f}")
    print(f"Profit Difference: {abs(results_lpa['optimal_solution'].profit - profit_pf):.4f}")

if __name__ == "__main__":
    compare_optimizations()
