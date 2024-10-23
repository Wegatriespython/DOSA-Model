import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from functools import partial

@dataclass
class OptimizationResult:
    labor: float
    production: float
    profit: float
    is_feasible: bool

def calculate_profit(labor: float, production: float, params: Dict) -> float:
    """Calculate profit for given labor and production levels"""
    revenue = production * params['price']
    labor_cost = labor * params['wage']
    # Fixed costs include capital costs
    fixed_costs = params['capital'] * params['capital_price'] * params['depreciation_rate']
    return revenue - labor_cost - fixed_costs

def production_constraint(labor: float, production: float, params: Dict) -> float:
    """Calculate production possibility given labor input with separate productivity parameters"""
    epsilon = 1e-6
    max_possible_production = (
        (params['labor_productivity'] * (labor + epsilon)) ** (1 - params['capital_elasticity']) * 
        (params['capital_productivity'] * params['capital']) ** params['capital_elasticity']
    )
    return max_possible_production - production

def optimize_labor_production(params: Dict) -> OptimizationResult:
    """Optimize labor and production for fixed capital"""
    model = pyo.ConcreteModel()
    
    # Variables
    model.labor = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, params['labor_supply']))
    model.production = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, params['demand']))
    
    # Objective
    model.objective = pyo.Objective(
        expr=lambda m: calculate_profit(m.labor, m.production, params),
        sense=pyo.maximize
    )
    
    # Production possibility constraint
    model.prod_constraint = pyo.Constraint(
        expr=lambda m: production_constraint(m.labor, m.production, params) >= 0
    )
    
    # Solve
    solver = SolverFactory('ipopt')
    results = solver.solve(model, tee=False)
    
    is_optimal = (results.solver.status == pyo.SolverStatus.ok and 
                 results.solver.termination_condition == pyo.TerminationCondition.optimal)
    
    return OptimizationResult(
        labor=pyo.value(model.labor),
        production=pyo.value(model.production),
        profit=pyo.value(model.objective),
        is_feasible=is_optimal
    )

def plot_profit_landscape(params: Dict, result: OptimizationResult) -> None:
    """Create 3D visualization of profit landscape"""
    labor_range = np.linspace(0, params['labor_supply'], 50)
    prod_range = np.linspace(0, params['demand'], 50)
    L, P = np.meshgrid(labor_range, prod_range)
    
    # Calculate profit for each point
    Z = np.zeros_like(L)
    for i in range(len(labor_range)):
        for j in range(len(prod_range)):
            Z[i,j] = calculate_profit(L[i,j], P[i,j], params)
            
            # Mask points that violate production constraint
            if production_constraint(L[i,j], P[i,j], params) < 0:
                Z[i,j] = np.nan
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(L, P, Z, cmap='viridis')
    
    # Plot optimal point
    ax.scatter([result.labor], [result.production], [result.profit], 
               color='red', s=100, label='Optimal Point')
    
    ax.set_xlabel('Labor')
    ax.set_ylabel('Production')
    ax.set_zlabel('Profit')
    plt.colorbar(surf)
    plt.title('Profit Landscape')
    plt.legend()
    plt.show()

def find_price_wage_envelope(params: Dict, 
                           optimal_result: OptimizationResult,
                           tolerance: float = 1e-4) -> Dict[str, float]:
    """Find minimum price and maximum wage that maintain optimal solution"""
    
    def binary_search(param_name: str, initial_value: float, 
                     direction: int, step: float = 0.1) -> float:
        current_value = initial_value
        params_copy = params.copy()
        
        while True:
            params_copy[param_name] = current_value
            result = optimize_labor_production(params_copy)
            
            # Check if solution is still close to original optimal
            is_similar = (
                abs(result.labor - optimal_result.labor) < tolerance and
                abs(result.production - optimal_result.production) < tolerance
            )
            
            if not is_similar:
                if step < tolerance:
                    return current_value - direction * step
                current_value -= direction * step
                step /= 2
            else:
                current_value += direction * step
    
    # Find minimum viable price (searching downward)
    min_price = binary_search('price', params['price'], -1)
    
    # Find maximum viable wage (searching upward)
    max_wage = binary_search('wage', params['wage'], 1)
    
    return {
        'minimum_viable_price': min_price,
        'maximum_viable_wage': max_wage,
        'price_margin': params['price'] - min_price,
        'wage_margin': max_wage - params['wage']
    }

def analyze_productivity_impact(base_params: Dict, n_points: int = 500) -> Dict[str, np.ndarray]:
    """Analyze how changing productivity affects optimal solutions"""
    productivity_range = np.linspace(1, 10, n_points)
    results = {
        'productivity': productivity_range,
        'profit': np.zeros(n_points),
        'labor': np.zeros(n_points),
        'production': np.zeros(n_points)
    }
    
    for i, prod in enumerate(productivity_range):
        params = base_params.copy()
        params['productivity'] = prod
        optimal = optimize_labor_production(params)
        
        if optimal.is_feasible:
            results['profit'][i] = optimal.profit
            results['labor'][i] = optimal.labor
            results['production'][i] = optimal.production
        else:
            results['profit'][i] = np.nan
            results['labor'][i] = np.nan
            results['production'][i] = np.nan
    
    return results

def plot_productivity_analysis(results: Dict[str, np.ndarray]) -> None:
    """Create visualization of how productivity affects key metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Impact of Productivity on Optimal Solutions')
    
    # Profit plot
    ax1.plot(results['productivity'], results['profit'], 'b-', label='Profit')
    ax1.set_ylabel('Profit')
    ax1.set_xlabel('Productivity')
    ax1.grid(True)
    ax1.legend()
    
    # Labor plot
    ax2.plot(results['productivity'], results['labor'], 'r-', label='Labor')
    ax2.set_ylabel('Labor')
    ax2.set_xlabel('Productivity')
    ax2.grid(True)
    ax2.legend()
    
    # Production plot
    ax3.plot(results['productivity'], results['production'], 'g-', label='Production')
    ax3.set_ylabel('Production')
    ax3.set_xlabel('Productivity')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def find_target_productivity_combinations(base_params: Dict, 
                                       target_labor: float = 6.0,
                                       target_production: float = 6.0,
                                       tolerance: float = 0.1,
                                       n_points: int = 50) -> Dict[str, np.ndarray]:
    """Find combinations of labor and capital productivity that yield target values"""
    # Create parameter grid
    A_L = np.linspace(0.1, 5, n_points)  # labor productivity range
    A_K = np.linspace(0.1, 5, n_points)  # capital productivity range
    A_L_grid, A_K_grid = np.meshgrid(A_L, A_K)
    
    results = {
        'labor_prod': A_L_grid,
        'capital_prod': A_K_grid,
        'labor_diff': np.zeros_like(A_L_grid),
        'prod_diff': np.zeros_like(A_L_grid),
        'is_target': np.zeros_like(A_L_grid, dtype=bool)
    }
    
    for i in range(n_points):
        for j in range(n_points):
            params = base_params.copy()
            params['labor_productivity'] = A_L_grid[i,j]
            params['capital_productivity'] = A_K_grid[i,j]
            
            optimal = optimize_labor_production(params)
            
            if optimal.is_feasible:
                results['labor_diff'][i,j] = abs(optimal.labor - target_labor)
                results['prod_diff'][i,j] = abs(optimal.production - target_production)
                results['is_target'][i,j] = (results['labor_diff'][i,j] < tolerance and 
                                           results['prod_diff'][i,j] < tolerance)
            else:
                results['labor_diff'][i,j] = np.nan
                results['prod_diff'][i,j] = np.nan
                results['is_target'][i,j] = False
    
    return results

def plot_productivity_combinations(results: Dict[str, np.ndarray]) -> None:
    """Visualize the combinations of productivity parameters"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot difference from targets
    total_diff = results['labor_diff'] + results['prod_diff']
    im1 = ax1.pcolormesh(results['labor_prod'], results['capital_prod'], 
                        total_diff, shading='auto', cmap='viridis')
    ax1.set_xlabel('Labor Productivity (A_L)')
    ax1.set_ylabel('Capital Productivity (A_K)')
    ax1.set_title('Distance from Target Values')
    plt.colorbar(im1, ax=ax1)
    
    # Plot target-meeting combinations
    ax2.scatter(results['labor_prod'][results['is_target']], 
                results['capital_prod'][results['is_target']], 
                c='red', s=20, alpha=0.6)
    ax2.set_xlabel('Labor Productivity (A_L)')
    ax2.set_ylabel('Capital Productivity (A_K)')
    ax2.set_title('Combinations Meeting Target Values')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_production_labor(params: Dict, 
                           find_target_combinations: bool = False,
                           **kwargs) -> Dict:
    """Main analysis function"""
    results = {}
    
    if find_target_combinations:
        productivity_results = find_target_productivity_combinations(params, **kwargs)
        plot_productivity_combinations(productivity_results)
        results['productivity_combinations'] = productivity_results
    
    return results

# Example usage:
if __name__ == "__main__":
    test_params = {
        'capital': 6,
        'price': 1,
        'wage': 1,
        'capital_price': 5,
        'depreciation_rate': 0.1,
        'labor_productivity': 1.0,  # Initial A_L
        'capital_productivity': 1.0,  # Initial A_K
        'capital_elasticity': 0.5,
        'labor_supply': 30,
        'demand': 30
    }
    
    # Run analysis to find productivity combinations
    results = analyze_production_labor(
        test_params, 
        find_target_combinations=True,
        target_labor=6.0,
        target_production=6.0,
        tolerance=0.1
    )
