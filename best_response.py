import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple

def worker_utility(params: Dict, x: np.ndarray) -> float:
    """
    Calculate worker's utility.
    
    :param params: Dictionary of parameters
    :param x: Array of decision variables [consumption, leisure]
    :return: Negative utility (for minimization)
    """
    c, l = x
    alpha = params['alpha']
    return -((c ** alpha) * (l ** (1 - alpha)))

def worker_constraints(params: Dict) -> List[Dict]:
    """
    Define constraints for worker's optimization problem.
    
    :param params: Dictionary of parameters
    :return: List of constraint dictionaries
    """
    def budget_constraint(x):
        c, l = x
        return params['wealth'] + params['wage'] * (params['max_hours'] - l) - params['price'] * c

    return [
        {'type': 'ineq', 'fun': budget_constraint},
        {'type': 'ineq', 'fun': lambda x: params['max_hours'] - x[1]},  # Time constraint
        {'type': 'ineq', 'fun': lambda x: x[0]},  # Non-negativity of consumption
        {'type': 'ineq', 'fun': lambda x: x[1]}   # Non-negativity of leisure
    ]

def firm_profit(params: Dict, x: np.ndarray) -> float:
    """
    Calculate firm's profit.
    
    :param params: Dictionary of parameters
    :param x: Array of decision variables [labor, capital]
    :return: Negative profit (for minimization)
    """
    L, K = x
    A = params['productivity']
    beta = params['capital_share']
    p = params['price']
    w = params['wage']
    r = params['capital_rental_rate']
    
    Y = A * (K ** beta) * (L ** (1 - beta))
    return -(p * Y - w * L - r * K)

def firm_constraints(params: Dict) -> List[Dict]:
    """
    Define constraints for firm's optimization problem.
    
    :param params: Dictionary of parameters
    :return: List of constraint dictionaries
    """
    return [
        {'type': 'ineq', 'fun': lambda x: x[0]},  # Non-negativity of labor
        {'type': 'ineq', 'fun': lambda x: x[1]}   # Non-negativity of capital
    ]

def best_response(agent_type: str, params: Dict) -> Tuple[np.ndarray, float]:
    """
    Compute the best response for a given agent type.
    
    :param agent_type: 'worker' or 'firm'
    :param params: Dictionary of parameters
    :return: Tuple of optimal decision variables and optimal value
    """
    if agent_type == 'worker':
        objective = lambda x: worker_utility(params, x)
        constraints = worker_constraints(params)
        initial_guess = [params['wealth'] / params['price'], params['max_hours'] / 2]
    elif agent_type == 'firm':
        objective = lambda x: firm_profit(params, x)
        constraints = firm_constraints(params)
        initial_guess = [10, 10]  # Arbitrary initial guess for labor and capital
    else:
        raise ValueError("Invalid agent type. Must be 'worker' or 'firm'.")

    result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    return result.x, -result.fun  # Return negative of objective because we minimized the negative utility/profit

def market_clearing_error(prices: np.ndarray, workers: List[Dict], firms: List[Dict]) -> np.ndarray:
    """
    Compute the market clearing error for both labor and goods markets.
    
    :param prices: Array of [wage, goods_price]
    :param workers: List of worker parameter dictionaries
    :param firms: List of firm parameter dictionaries
    :return: Array of market clearing errors [labor_error, goods_error]
    """
    wage, goods_price = prices
    
    labor_supply = 0
    goods_demand = 0
    for worker in workers:
        worker['wage'] = wage
        worker['price'] = goods_price
        worker_decision, _ = best_response('worker', worker)
        labor_supply += worker['max_hours'] - worker_decision[1]  # working hours
        goods_demand += worker_decision[0]  # consumption
    
    labor_demand = 0
    goods_supply = 0
    for firm in firms:
        firm['wage'] = wage
        firm['price'] = goods_price
        firm_decision, _ = best_response('firm', firm)
        labor_demand += firm_decision[0]
        goods_supply += firm['productivity'] * (firm_decision[1] ** firm['capital_share']) * (firm_decision[0] ** (1 - firm['capital_share']))
    
    return np.array([labor_supply - labor_demand, goods_demand - goods_supply])

def find_equilibrium(workers: List[Dict], firms: List[Dict], initial_prices: np.ndarray) -> np.ndarray:
    """
    Find the equilibrium prices that clear both markets.
    
    :param workers: List of worker parameter dictionaries
    :param firms: List of firm parameter dictionaries
    :param initial_prices: Initial guess for [wage, goods_price]
    :return: Equilibrium prices [wage, goods_price]
    """
    result = minimize(
        lambda prices: np.sum(market_clearing_error(prices, workers, firms)**2),
        initial_prices,
        method='Nelder-Mead'
    )
    
    if not result.success:
        raise ValueError(f"Equilibrium finding failed: {result.message}")
    
    return result.x

# Example usage
if __name__ == "__main__":
    # Example parameters
    workers = [
        {'alpha': 0.7, 'max_hours': 24, 'wealth': 100} for _ in range(10)
    ]
    firms = [
        {'productivity': 10, 'capital_share': 0.3, 'capital_rental_rate': 0.05} for _ in range(5)
    ]
    
    initial_prices = np.array([10, 1])  # Initial guess for [wage, goods_price]
    
    equilibrium_prices = find_equilibrium(workers, firms, initial_prices)
    print(f"Equilibrium wage: {equilibrium_prices[0]:.2f}")
    print(f"Equilibrium goods price: {equilibrium_prices[1]:.2f}")
    
    # Compute and print market clearing error at equilibrium
    error = market_clearing_error(equilibrium_prices, workers, firms)
    print(f"Labor market clearing error: {error[0]:.2e}")
    print(f"Goods market clearing error: {error[1]:.2e}")
