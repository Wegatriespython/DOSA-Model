import numpy as np
from scipy.optimize import minimize
from Config import config

from typing import List

import numpy as np
from scipy.optimize import minimize

def neoclassical_profit_maximization(
        current_capital: float,
        current_labor: float,
        current_price: float,
        current_productivity: float,
        expected_demand: list,
        expected_price: list,
        capital_price: float,
        capital_elasticity: float,
        current_inventory: float,
        depreciation_rate: float,
        expected_periods: int,
        discount_rate: float
    ):
    def objective(x):
        labor, capital, production = x
        total_discounted_profit = 0
        inventory = current_inventory
        
        for t in range(expected_periods):
            price = max(expected_price[t] if t < len(expected_price) else expected_price[-1], 1e-10)
            demand = max(expected_demand[t] if t < len(expected_demand) else expected_demand[-1], 1e-10)
            
            # Calculate price elasticity of demand with error handling
            if t > 0:
                prev_price = max(expected_price[t-1] if t < len(expected_price) else expected_price[-2], 1e-10)
                prev_demand = max(expected_demand[t-1] if t < len(expected_demand) else expected_demand[-2], 1e-10)
                price_diff = price - prev_price
                demand_diff = demand - prev_demand
                if abs(price_diff) < 1e-10 or abs(prev_demand) < 1e-10:
                    price_elasticity = -1  # Default to unit elasticity if division is unsafe
                else:
                    price_elasticity = (demand_diff / prev_demand) / (price_diff / prev_price)
            else:
                price_elasticity = -1  # Assume unit elasticity for the first period
            
            # Ensure elasticity is within reasonable bounds
            price_elasticity = np.clip(price_elasticity, -10, -0.1)
            
            # Adjust production based on elasticity
            elastic_adjustment = 1 / (1 + abs(price_elasticity))
            adjusted_production = production * elastic_adjustment
            
            sales = min(adjusted_production + inventory, demand)
            revenue = price * sales
            labor_cost = labor
            capital_cost = (capital - current_capital) * capital_price if t == 0 else 0
            inventory_cost = depreciation_rate * price * max(inventory + adjusted_production - sales, 0)
            
            period_profit = revenue - labor_cost - capital_cost - inventory_cost
            total_discounted_profit += period_profit / ((1 + max(discount_rate, 1e-10)) ** t)
            
            inventory = max(inventory + adjusted_production - sales, 0)

        return -total_discounted_profit  # Negative because we're minimizing

    # Relaxed constraints
    def constraint(x):
        labor, capital, production = x
        return production - 0.5 * current_productivity * (capital ** capital_elasticity) * (labor ** (1 - capital_elasticity))

    max_labor = max(1000, current_labor * 2)
    max_capital = max(current_capital * 2, current_capital + max_labor / max(capital_price, 1))
    max_production = 2 * current_productivity * (max_capital ** capital_elasticity) * (max_labor ** (1 - capital_elasticity))
    
    bounds = [(0, max_labor), (current_capital * 0.5, max_capital), (0, max_production)]
    constraints = {'type': 'ineq', 'fun': constraint}

    initial_guess = [current_labor, current_capital, current_productivity * (current_capital ** capital_elasticity) * (current_labor ** (1 - capital_elasticity))]


    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    
    optimal_labor = max(1, round(result.x[0]))
    optimal_capital = max(current_capital * 0.5, round(result.x[1]))
    optimal_production = max(1, round(result.x[2]))


    optimal_price = expected_price[0] if expected_price else current_price

    return optimal_labor, optimal_capital, optimal_price, optimal_production

# Example usage remains the same





def cobb_douglas_production(productivity, capital, labor, capital_elasticity):
    return productivity * (capital ** capital_elasticity) * (labor ** (1 - capital_elasticity))

    """
    Adjust price based on sales vs expected demand.
    
    :param current_price: The current price of the good
    :param sales: Actual sales in the last period
    :param expected_demand: Expected demand for the last period
    :return: Adjusted price
    """
    if sales >= expected_demand:
        return current_price * 1.05  # Increase price by 5%
    else:
        return current_price * 0.95  # Decrease price by 5%