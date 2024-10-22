import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Union

def validate_params(params: Dict) -> bool:
    """Validates input parameters and their relationships"""
    required_keys = ['current_capital', 'current_labor', 'current_price', 
                    'current_productivity', 'expected_demand', 'expected_price',
                    'capital_price', 'capital_elasticity', 'current_inventory',
                    'depreciation_rate', 'expected_periods', 'discount_rate',
                    'budget', 'wage', 'capital_supply', 'labor_supply']
    
    # First check if all required keys exist
    if not all(key in params for key in required_keys):
        print("Missing required parameters")
        return False

    # Check list parameters
    list_params = {'wage', 'labor_supply', 'expected_demand', 'expected_price'}
    for key in list_params:
        if not isinstance(params[key], (list, np.ndarray)) or not params[key]:
            print(f"Parameter {key} must be a non-empty list")
            return False

    # Check numeric parameters
    numeric_params = {
        'current_capital', 'current_labor', 'current_price', 
        'current_productivity', 'capital_price', 'capital_elasticity',
        'current_inventory', 'depreciation_rate', 'expected_periods',
        'discount_rate', 'budget', 'capital_supply'
    }
    
    for key in numeric_params:
        if not isinstance(params[key], (int, float)) or params[key] < 0:
            print(f"Parameter {key} must be a non-negative number")
            return False

    # Check length consistency
    if len(params['expected_demand']) != params['expected_periods']:
        print("Length of expected_demand must match expected_periods")
        return False

    return True

def calculate_scaling_factors(guess_capital: float, guess_labor: float, 
                            expected_price: List[float], expected_demand: List[float]) -> Dict[str, float]:
    """Compute scaling factors for numerical stability"""
    return {
        'capital': max(1.0, guess_capital),
        'labor': max(1.0, guess_labor),
        'price': max(1.0, max(expected_price)),
        'demand': max(1.0, max(expected_demand))
    }

def setup_solver_options(solver: pyo.SolverFactory, linear_solver: str = 'mumps') -> None:
    """Configure solver options for better convergence"""
    solver.options.update({
        'max_iter': 10000,
        'tol': 1e-3,
        'halt_on_ampl_error': 'yes',
        'linear_solver': linear_solver,
        'warm_start_init_point': 'yes',
        'mu_strategy': 'adaptive',
        'print_level': 1,
        'linear_scaling_on_demand': 'yes'
    })

def profit_maximization(params: Dict) -> Optional[Dict]:
    """Main entry point for profit maximization"""
    try:
        if not validate_params(params):
            return None

        # Preprocess parameters - handle lists safely
        labor_supply = float(params['labor_supply'][0]) /16
        wage = float(params['wage'][0])/16
        expected_demand = list(np.array(params['expected_demand'])/5)  # Create a copy
        
        return _profit_maximization(
            current_capital=params['current_capital'],
            current_labor=params['current_labor'],
            current_price=params['current_price'],
            current_productivity=params['current_productivity'],
            expected_demand=expected_demand,
            expected_price=params['expected_price'],
            capital_price=params['capital_price'],
            capital_elasticity=params['capital_elasticity'],
            current_inventory=params['current_inventory'],
            depreciation_rate=params['depreciation_rate'],
            expected_periods=params['expected_periods'],
            discount_rate=params['discount_rate'],
            budget=params['budget'],
            wage=wage,
            capital_supply=params['capital_supply'],
            labor_supply=labor_supply
        )
    except Exception as e:
        print(f"Error in profit_maximization: {str(e)}")
        return None

def _profit_maximization(
        current_capital: float, current_labor: float, current_price: float, 
        current_productivity: float, expected_demand: List[float], 
        expected_price: List[float], capital_price: float, capital_elasticity: float,
        current_inventory: float, depreciation_rate: float, expected_periods: int, 
        discount_rate: float, budget: float, wage: float, 
        capital_supply: float, labor_supply: float, 
        linear_solver: str = 'mumps') -> Optional[Dict]:
    """Core optimization logic with improved numerical stability"""
    
    model = pyo.ConcreteModel()
    
    # Compute initial guesses and bounds
    max_labor = labor_supply
    max_capital = current_capital + capital_supply
    guess_labor = max(1e-6, min(budget/wage, max_labor/2))  # Improved initial guess
    guess_capital = current_capital  # Conservative initial guess
    
    # Calculate scaling factors for better numerical stability
    scale = {
        'capital': max(1.0, max_capital),
        'labor': max(1.0, max_labor),
        'production': max(1.0, max(expected_demand)),
        'price': max(1.0, max(expected_price))
    }
    
    # Model setup
    model.T = pyo.RangeSet(0, expected_periods - 1)
    
    # Variables with scaling
    model.labor = pyo.Var(model.T, domain=pyo.NonNegativeReals, 
                         initialize=guess_labor/scale['labor'],
                         bounds=(0.0, max_labor/scale['labor']))
    
    model.capital = pyo.Var(model.T, domain=pyo.NonNegativeReals,
                           initialize=guess_capital/scale['capital'],
                           bounds=(current_capital/scale['capital'], 
                                 max_capital/scale['capital']))
    
    model.production = pyo.Var(model.T, domain=pyo.NonNegativeReals, 
                              initialize=1.0/scale['production'])
    model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, 
                             initialize=current_inventory/scale['production'])
    model.sales = pyo.Var(model.T, domain=pyo.NonNegativeReals, 
                         initialize=1.0/scale['production'])

    # Objective with scaling
    def objective_rule(model):
        return sum(
            (expected_price[t] * model.sales[t] * scale['production']
             - wage * model.labor[t] * scale['labor']
             - capital_price * (model.capital[t] * scale['capital'] - current_capital) 
             * (1 if t == 0 else 0)
            ) / ((1 + discount_rate) ** t)
            for t in model.T
        )
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Production constraint with scaling
    def production_constraint_rule(model, t):
        epsilon = 1e-6
        return model.production[t] * scale['production'] == current_productivity * \
               ((model.labor[t] * scale['labor'] + epsilon) ** (1 - capital_elasticity)) * \
               ((model.capital[t] * scale['capital'] + epsilon) ** capital_elasticity)
    model.production_constraint = pyo.Constraint(model.T, rule=production_constraint_rule)

    # Budget constraint with scaling
    def budget_constraint_rule(model, t):
        if t == 0:
            return (wage * model.labor[t] * scale['labor'] + 
                   capital_price * (model.capital[t] * scale['capital'] - current_capital) <= budget)
        return wage * model.labor[t] * scale['labor'] <= budget
    model.budget_constraint = pyo.Constraint(model.T, rule=budget_constraint_rule)

    # Inventory balance with scaling
    def inventory_balance_rule(model, t):
        if t == 0:
            return model.inventory[t] * scale['production'] == \
                   current_inventory + model.production[t] * scale['production'] - \
                   model.sales[t] * scale['production']
        else:
            return model.inventory[t] * scale['production'] == \
                   model.inventory[t-1] * scale['production'] + \
                   model.production[t] * scale['production'] - \
                   model.sales[t] * scale['production']
    model.inventory_balance = pyo.Constraint(model.T, rule=inventory_balance_rule)

    # Sales constraints with scaling
    def sales_constraint_demand_rule(model, t):
        return model.sales[t] * scale['production'] <= expected_demand[t]
    model.sales_constraint_demand = pyo.Constraint(model.T, rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model, t):
        return model.sales[t] * scale['production'] <= \
               model.inventory[t] * scale['production'] + \
               model.production[t] * scale['production']
    model.sales_constraint_inventory = pyo.Constraint(model.T, rule=sales_constraint_inventory_rule)

    # Solve
    solver = SolverFactory('ipopt')
    setup_solver_options(solver, linear_solver)
    results = solver.solve(model, tee=False)
    
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        
        # Verify budget constraint with scaled values
        total_cost = sum(wage * pyo.value(model.labor[t]) * scale['labor'] + 
                        (pyo.value(model.capital[t]) * scale['capital'] - current_capital) 
                        * capital_price * (1 if t == 0 else 0) for t in model.T)
        
        if total_cost > budget * 1.0000001:
            print(f"WARNING: Budget constraint violated. Total cost: {total_cost}, Budget: {budget}")
            return None
            
        return round_results({
            'optimal_labor': pyo.value(model.labor[0]) * scale['labor'],
            'optimal_capital': pyo.value(model.capital[0]) * scale['capital'],
            'optimal_production': pyo.value(model.production[0]) * scale['production'],
            'optimal_price': expected_price[0],
            'optimal_sales': [pyo.value(model.sales[t]) * scale['production'] for t in model.T],
            'optimal_inventory': [pyo.value(model.inventory[t]) * scale['production'] for t in model.T]
        })
    
    print(f"Optimization failed. Status: {results.solver.status}, "
          f"Termination: {results.solver.termination_condition}")
    return None

def round_results(results):
    rounded = {}
    for key, value in results.items():
        if isinstance(value, list):
            rounded[key] = [round(v) for v in value]
        else:
            rounded[key] = round(value)
    return rounded
