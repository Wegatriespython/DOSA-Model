import numpy as np
from julia import Main

# Load the Julia function
Main.include(r"V:\Python Port\Src\Utilities\Profit_maximization.jl")

def profit_maximization(
    current_capital, current_labor, current_price, current_productivity,
    expected_demand, expected_price, capital_price, capital_elasticity,
    current_inventory, depreciation_rate, expected_periods, discount_rate,
    budget, wage, capital_supply, labor_supply):

    # Convert numpy arrays to Julia arrays
    expected_demand = Main.Array(expected_demand)
    expected_price = Main.Array(expected_price)

    # Call the Julia function
    result = Main.profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage, capital_supply, labor_supply
    )

    if result is not None:

        return {k: np.array(v) if isinstance(v, list) else v for k, v in result.items()}
    else:
        return None

# The rest of your Python code remains the same
