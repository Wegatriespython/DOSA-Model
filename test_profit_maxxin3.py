import numpy as np
from Simple_profit_maxxing import profit_maximization

def test_profit_maximization_budget_constraint():
    # Test cases based on the observed inputs
    test_cases = [
        {
            "name": "Firm2 Initial",
            "params": {
                "current_capital": 5,
                "current_labor": 0,
                "current_price": 1,
                "current_productivity": 1,
                "expected_demand": [30]*30,
                "expected_price": [1]*30,
                "capital_price": 3,
                "capital_elasticity": 0.3,
                "current_inventory": 0,
                "depreciation_rate": 0.1,
                "expected_periods": 30,
                "discount_rate": 0.05,
                "budget": 4.6,
                "wage": 1
            }
        },
        {
            "name": "Firm1 Initial",
            "params": {
                "current_capital": 20,
                "current_labor": 0,
                "current_price": 3,
                "current_productivity": 1,
                "expected_demand": [4.6]*30,
                "expected_price": [3]*30,
                "capital_price": 3,
                "capital_elasticity": 0.3,
                "current_inventory": 3.6,
                "depreciation_rate": 0.1,
                "expected_periods": 30,
                "discount_rate": 0.05,
                "budget": 19.6,
                "wage": 1
            }
        }
    ]
    test_cases.append({
           "name": "Error Scenario",
           "params": {
               "current_capital": 20,
               "current_labor": 1,
               "current_price": 3.0,
               "current_productivity": 1.005,
               "expected_demand": [1.4150212035674503] * 30,
               "expected_price": [3.0] * 30,
               "capital_price": 3.0,
               "capital_elasticity": 0.3,
               "current_inventory": 1.9131876,
               "depreciation_rate": 0.1,
               "expected_periods": 30,
               "discount_rate": 0.05,
               "budget": 7.224719196743946,
               "wage": 5.314843603256058
           }
       })

    for case in test_cases:
        print(f"\nTesting {case['name']}:")
        result = profit_maximization(**case['params'])

        if result is None:
            print("Optimization failed")
            continue

        print("Optimization results:")
        for key, value in result.items():
            print(f"{key}: {value}")

        # Validate results
        validate_results(case['params'], result)

def validate_results(params, result):
    # Check if labor and capital are within reasonable bounds
    max_labor = params['budget'] / params['wage']
    max_capital = params['budget'] / params['capital_price'] + params['current_capital']

    if result['optimal_labor'] > max_labor:
        print(f"WARNING: Optimal labor ({result['optimal_labor']}) exceeds maximum possible ({max_labor})")

    if result['optimal_capital'] > max_capital:
        print(f"WARNING: Optimal capital ({result['optimal_capital']}) exceeds maximum possible ({max_capital})")

    # Check if production is consistent with inputs
    expected_production = params['current_productivity'] * (result['optimal_capital'] ** params['capital_elasticity']) * (result['optimal_labor'] ** (1 - params['capital_elasticity']))
    if abs(result['optimal_production'] - expected_production) > 0.01 * expected_production:
        print(f"WARNING: Optimal production ({result['optimal_production']}) is inconsistent with inputs (expected: {expected_production})")

    # Check if sales do not exceed production + inventory
    total_available = result['optimal_production'] + params['current_inventory']
    if any(sale > total_available for sale in result['optimal_sales']):
        print(f"WARNING: Some optimal sales exceed total available goods ({total_available})")

    # Check if total cost does not exceed budget
    total_cost = result['optimal_labor'] * params['wage'] + (result['optimal_capital'] - params['current_capital']) * params['capital_price']
    if total_cost > params['budget']:
        print(f"WARNING: Total cost ({total_cost}) exceeds budget ({params['budget']})")
        # Additional validation for the new error scenario
    if result is not None:
        if result['optimal_capital'] <= 0:
            print(f"WARNING: Optimal capital ({result['optimal_capital']}) is non-positive")
        if result['optimal_labor'] <= 0:
            print(f"WARNING: Optimal labor ({result['optimal_labor']}) is non-positive")
if __name__ == "__main__":
    test_profit_maximization_budget_constraint()
