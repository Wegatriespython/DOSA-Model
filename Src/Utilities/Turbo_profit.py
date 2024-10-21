import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from functools import lru_cache



def profit_maximization(params):

    current_capital = params['current_capital']
    current_labor = params['current_labor']
    current_price = params['current_price']
    current_productivity = params['current_productivity']
    expected_demand = [demand/5 for demand in params['expected_demand']]
    expected_price = params['expected_price']
    capital_price = params['capital_price']
    capital_elasticity = params['capital_elasticity']
    current_inventory = params['current_inventory']
    depreciation_rate = params['depreciation_rate']
    expected_periods = params['expected_periods']
    discount_rate = params['discount_rate']
    budget = params['budget']
    wage = params['wage'][0]
    capital_supply = params['capital_supply']
    labor_supply = [supply / (16*5) for supply in params['labor_supply']][0]


    results = _profit_maximization(current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage,capital_supply, labor_supply, linear_solver = 'mumps')
    if results is None:
        print("No optimal solution found")
        return None
    else:
        return round_results(results)


def _profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage,capital_supply, labor_supply, linear_solver = 'mumps') :

    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, expected_periods - 1)
    max_labor = max(labor_supply + current_labor, 3)
    print(f"max_labor: {max_labor}")
    max_capital = capital_supply + current_capital


    guess_capital = (current_capital + max_capital)/2
    guess_labor = (current_labor + max_labor)/2

    # Scaling factors
    scale_capital = max(1, guess_capital)
    scale_labor = max(1, guess_labor)
    print(f"scale_labor: {scale_labor}")
    scale_price = max(1, max(expected_price))
    scale_demand = max(1, max(expected_demand))

    # Variables with scaling and lower bounds
    model.labor = pyo.Var(domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_labor), bounds=(1e-6, max_labor))
    model.capital = pyo.Var(domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_capital), bounds=(1e-6, max_capital))
    model.production = pyo.Var(domain=pyo.NonNegativeReals, initialize=1e-6)
    model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, current_inventory))
    model.sales = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1e-6)

    # Objective
    def objective_rule(model):
        obj_value = sum(
            (expected_price[t] * model.sales[t] / scale_price
             - wage * model.labor / scale_labor
             - (model.capital - current_capital) * capital_price / scale_capital * (1 if t == 0 else 0)
             - depreciation_rate * expected_price[t] * model.inventory[t] / scale_price
            ) / ((1 + discount_rate) ** t)
            for t in model.T
        )
        return obj_value
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def production_constraint_rule(model):
        return model.production == current_productivity * (model.capital ** capital_elasticity) * (model.labor ** (1 - capital_elasticity))
    model.production_constraint = pyo.Constraint(rule=production_constraint_rule)

    def budget_constraint_rule(model):
        return (wage * model.labor + (model.capital - current_capital) * capital_price <= budget * 1.0000001) #allow 0.0001% slack
    model.budget_constraint = pyo.Constraint(rule=budget_constraint_rule)

    def inventory_balance_rule(model, t):
        if t == 0:
            return model.inventory[t] == current_inventory + model.production - model.sales[t]
        else:
            return model.inventory[t] == model.inventory[t-1] + model.production - model.sales[t]
    model.inventory_balance = pyo.Constraint(model.T, rule=inventory_balance_rule)

    def sales_constraint_demand_rule(model, t):
        return model.sales[t] <= expected_demand[t]
    model.sales_constraint_demand = pyo.Constraint(model.T, rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model, t):
        return model.sales[t] <= model.inventory[t] + model.production
    model.sales_constraint_inventory = pyo.Constraint(model.T, rule=sales_constraint_inventory_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-3  # Sets the convergence tolerance for the optimization algorithm
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver

    solver.options['warm_start_init_point'] = 'yes'  # Use warm start


    solver.options['mu_strategy'] = 'adaptive'
    solver.options['print_level'] = 1  # Increase print level for more information


    solver.options['linear_scaling_on_demand'] = 'yes'  # Perform linear scaling only when needed

    results = solver.solve(model, tee=False)



    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        total_cost = pyo.value(wage * model.labor + (model.capital - current_capital) * capital_price)
        if total_cost > budget * 1.0000001:  # Allow for 0.0001% violation due to numerical issues
            print(f"WARNING: Budget constraint violated. Total cost: {total_cost}, Budget: {budget}")

    # Check if the solver found an optimal solution
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        unrounded_results = {
            'optimal_labor': pyo.value(model.labor),
            'optimal_capital': pyo.value(model.capital),
            'optimal_production': pyo.value(model.production),
            'optimal_price': expected_price[0],
            'optimal_sales': [pyo.value(model.sales[t]) for t in model.T],
            'optimal_inventory': [pyo.value(model.inventory[t]) for t in model.T]
        }
        return round_results(unrounded_results)
    else:
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        return None
def round_results(results):
    rounded = {}
    for key, value in results.items():
        if isinstance(value, list):
            rounded[key] = [round(v) for v in value]
        else:
            rounded[key] = round(value)
    return rounded
