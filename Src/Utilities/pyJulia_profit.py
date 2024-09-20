import numpy as np
from julia import Main
from julia import Julia



julia = Julia(compiled_modules=False)
# Load the Julia function
Main.include(r"V:\Python Port\Src\Utilities\Continous_Profit_Maxxin.jl")

def profit_maximization(
    current_capital, current_labor, current_price, current_productivity,
    expected_demand, expected_price, capital_price, capital_elasticity,
    current_inventory, depreciation_rate, expected_periods, discount_rate,
    budget, wage, capital_supply, labor_supply, debt, carbon_intensity, new_carbon_intensity, carbon_tax, holding_cost):

    # Convert numpy arrays to Julia arrays
    expected_demand = Main.Array(expected_demand)
    expected_price = Main.Array(expected_price)

    # Call the Julia function
    result, zero_profit_conditions = Main.profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage, capital_supply, labor_supply, debt, carbon_intensity, new_carbon_intensity, carbon_tax, holding_cost
    )

    if result is not None:

        return {k: np.array(v) if isinstance(v, list) else v for k, v in result.items()}
    else:
        return None

# The rest of your Python code remains the same
