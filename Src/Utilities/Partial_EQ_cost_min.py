import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from pyomo.util.infeasible import log_infeasible_constraints

def fake_result(params, profit_max_result):
    current_price = params['current_price']
    capital_price = params['capital_price']
    wage = params['wage']

    result = {
        'price': current_price * 0.75,
        'wage': wage *1.25,
        'capital_price': capital_price
    }
    return result

def cost_minimization(profit_max_result, params):
    sales = profit_max_result['optimal_sales'][0]

    if sales != 0:
        return separate_optimization(profit_max_result, params)
    else:
        result = fake_result(params, profit_max_result)
        return result, False

def _solve_for_price(profit_max_result, params):
    model = pyo.ConcreteModel()

    # Variable: price
    model.price = pyo.Var(domain=pyo.NonNegativeReals)

    # Fixed parameters
    wage = params['wage']
    capital_price = params['capital_price']
    depreciation_rate = params['depreciation_rate']
    holding_costs = params['holding_costs']
    carbon_tax_rate = params['carbon_tax_rate']

    # Fixed quantities from profit_max_result (first period only)
    model.sales = pyo.Param(initialize=profit_max_result['optimal_sales'][0])
    model.labor = pyo.Param(initialize=profit_max_result['optimal_labor'][0])
    model.capital = pyo.Param(initialize=profit_max_result['optimal_capital'][0])
    model.inventory = pyo.Param(initialize=profit_max_result['optimal_inventory'][0])
    model.emissions = pyo.Param(initialize=profit_max_result['optimal_emissions'][0])

    # Objective: Minimize price
    model.objective = pyo.Objective(expr=model.price * model.sales, sense=pyo.minimize)

    # Zero profit constraint: revenue >= costs
    def zero_profit_rule(model):
        revenue = model.price * model.sales
        costs = wage * model.labor
        return revenue >= costs
    model.zero_profit = pyo.Constraint(rule=zero_profit_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    results = solver.solve(model, tee=False)

    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        reservation_price = pyo.value(model.price)
        return {'price': reservation_price}, True
    else:
        print("Price optimization solver failed to find an optimal solution.")
        result = fake_result(params, profit_max_result)
        return result, False

def _solve_for_wage(profit_max_result, params):
    model = pyo.ConcreteModel()

    # Variable: wage
    model.wage = pyo.Var(domain=pyo.NonNegativeReals, initialize=params['wage'])

    # Fixed parameters
    capital_price = params['capital_price']
    depreciation_rate = params['depreciation_rate']
    holding_costs = params['holding_costs']
    carbon_tax_rate = params['carbon_tax_rate']

    # Handle current_price which may be a list
    price = params['current_price'][0] if isinstance(params['current_price'], (list, np.ndarray)) else params['current_price']

    # Fixed quantities from profit_max_result (first period only)
    model.price = pyo.Param(initialize=price)
    model.sales = pyo.Param(initialize=profit_max_result['optimal_sales'][0])
    model.labor = pyo.Param(initialize=profit_max_result['optimal_labor'][0])
    model.capital = pyo.Param(initialize=profit_max_result['optimal_capital'][0])
    model.inventory = pyo.Param(initialize=profit_max_result['optimal_inventory'][0])
    model.emissions = pyo.Param(initialize=profit_max_result['optimal_emissions'][0])

    # Objective: Maximize wage
    model.objective = pyo.Objective(expr=model.wage, sense=pyo.maximize)

    # Zero-profit constraint: revenue >= total costs
    def zero_profit_rule(model):
        revenue = model.price * model.sales
        costs = model.wage * model.labor
        return revenue >= costs
    model.zero_profit = pyo.Constraint(rule=zero_profit_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    results = solver.solve(model, tee=False)

    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        reservation_wage = pyo.value(model.wage)
        
        return {'wage': reservation_wage}, True
    else:
        print("Wage optimization solver failed to find an optimal solution.")
        result = fake_result(params, profit_max_result)
        return result, False

def separate_optimization(profit_max_result, params):
    # Step 1: Optimize for product price with wage held fixed
    if profit_max_result['optimal_sales'][0] == 0:
        result = fake_result(params, profit_max_result)
        return result, False
    if profit_max_result['optimal_labor'][0] == 0:
        result = fake_result(params, profit_max_result)
        return result, False
    result_price, _ = _solve_for_price(profit_max_result, params)
    new_price = result_price['price']

    # Step 2: Optimize for wage with product price held fixed
    params_with_new_price = params.copy()
    params_with_new_price['current_price'] = new_price
    result_wage, _ = _solve_for_wage(profit_max_result, params_with_new_price)
    new_wage = result_wage['wage']

    print(f"New wage: {new_wage}, New price: {new_price}")
    return {
        'price': new_price,
        'wage': new_wage,
        'capital_price': params['capital_price']
    }, True

    max_iterations = 100
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        # Step 1: Optimize for product price with wage held fixed
        result_price, price_success = _solve_for_price(profit_max_result, params)
        new_price = result_price['price']

        # Step 2: Optimize for wage with product price held fixed
        params_with_new_price = params.copy()
        params_with_new_price['current_price'] = new_price
        result_wage, wage_success = _solve_for_wage(profit_max_result, params_with_new_price)
        new_wage = result_wage['wage']

        # Check for convergence
        price_change = abs(new_price - params['current_price']) / params['current_price']
        wage_change = abs(new_wage - params['wage']) / params['wage']

        print(f"Iteration {iteration + 1}: New wage: {new_wage}, New price: {new_price}")

        if price_change < tolerance and wage_change < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

        # Update params for next iteration
        params['current_price'] = new_price
        params['wage'] = new_wage

    else:
        print(f"Failed to converge after {max_iterations} iterations.")

    return {
        'price': new_price,
        'wage': new_wage,
        'capital_price': params['capital_price']
    }, price_success and wage_success
