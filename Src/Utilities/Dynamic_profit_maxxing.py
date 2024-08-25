import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

def profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, horizon, discount_rate,
        budget, wage, previous_labor, previous_capital, adjustment_cost_factor,
        linear_solver='ma27'):

    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, horizon - 1)

    max_labor = budget/max(1e-6,wage) + current_labor
    max_capital = budget/max(1e-6,capital_price) + current_capital

    guess_capital = (current_capital + max_capital)/2
    guess_labor = (current_labor + max_labor)/2

    # Scaling factors
    scale_capital = max(1, guess_capital)
    scale_labor = max(1, guess_labor)
    scale_price = max(1, current_price)
    scale_demand = max(1, max(expected_demand))

    # Variables with scaling and lower bounds
    model.labor = pyo.Var(domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_labor), bounds=(1e-6, None))
    model.capital = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_capital), bounds=(1e-6, None))
    model.production = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1e-6)
    model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, current_inventory))
    model.sales = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1e-6)
    model.investment = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1e-6)
    model.labor_change = pyo.Var(domain=pyo.Reals, initialize=0)
    model.capital_change = pyo.Var(model.T, domain=pyo.Reals, initialize=0)
    # Objective
    # Objective function
    def objective_rule(model):
        return sum(
            (expected_price[t] * model.sales[t] / scale_price
             - wage * model.labor / scale_labor
             - capital_price * model.investment[t] / scale_capital
             - depreciation_rate * expected_price[t] * model.capital[t] / scale_price
             - adjustment_cost_factor * (model.labor_change**2 + model.capital_change[t]**2)
            ) / ((1 + discount_rate) ** t)
            for t in model.T
        )
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def production_constraint_rule(model, t):
        return model.production[t] == current_productivity * (model.capital[t] ** capital_elasticity) * (model.labor ** (1 - capital_elasticity))
    model.production_constraint = pyo.Constraint(model.T, rule=production_constraint_rule)

    def capital_balance_rule(model, t):
        if t == 0:
            return model.capital[t] == current_capital + model.investment[t] - depreciation_rate * current_capital
        else:
            return model.capital[t] == model.capital[t-1] + model.investment[t] - depreciation_rate * model.capital[t-1]
    model.capital_balance = pyo.Constraint(model.T, rule=capital_balance_rule)

    def budget_constraint_rule(model):
        return (wage * model.labor + model.investment[0] * capital_price <= budget * 1.0000001) #allow 0.0001% slack
    model.budget_constraint = pyo.Constraint(rule=budget_constraint_rule)

    def inventory_balance_rule(model, t):
        if t == 0:
            return model.inventory[t] == current_inventory + model.production[t] - model.sales[t]
        else:
            return model.inventory[t] == model.inventory[t-1] + model.production[t] - model.sales[t]
    model.inventory_balance = pyo.Constraint(model.T, rule=inventory_balance_rule)

    def sales_constraint_demand_rule(model, t):
        return model.sales[t] <= expected_demand[t]
    model.sales_constraint_demand = pyo.Constraint(model.T, rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model, t):
        return model.sales[t] <= model.inventory[t] + model.production[t]
    model.sales_constraint_inventory = pyo.Constraint(model.T, rule=sales_constraint_inventory_rule)
    def labor_change_constraint(model):
        return model.labor_change == model.labor - previous_labor
    model.labor_change_constraint = pyo.Constraint(rule=labor_change_constraint)

    def capital_change_constraint(model, t):
        if t == 0:
            return model.capital_change[t] == model.capital[t] - previous_capital
        else:
            return model.capital_change[t] == model.capital[t] - model.capital[t-1]
    model.capital_change_constraint = pyo.Constraint(model.T, rule=capital_change_constraint)
    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        total_cost = pyo.value(wage * model.labor + model.investment[0] * capital_price)
        if total_cost > budget * 1.0000001:  # Allow for 0.0001% violation due to numerical issues
            print(f"WARNING: Budget constraint violated. Total cost: {total_cost}, Budget: {budget}")

    # Check if the solver found an optimal solution
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        return {
            'optimal_labor': pyo.value(model.labor),
            'optimal_capital': [pyo.value(model.capital[t]) for t in model.T],
            'optimal_investment': [pyo.value(model.investment[t]) for t in model.T],
            'optimal_production': [pyo.value(model.production[t]) for t in model.T],
            'optimal_price': expected_price[0],
            'optimal_sales': [pyo.value(model.sales[t]) for t in model.T],
            'optimal_inventory': [pyo.value(model.inventory[t]) for t in model.T]
        }
    else:
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        return None
