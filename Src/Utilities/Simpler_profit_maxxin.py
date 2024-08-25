import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

def profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage, linear_solver='ma27'):

    model = pyo.ConcreteModel()
    max_labor = budget/max(1e-6,wage) + current_labor
    max_capital = budget/max(1e-6,capital_price) + current_capital

    guess_capital = (current_capital + max_capital)/2
    guess_labor = (current_labor + max_labor)/2
    # Scaling factors
    scale_capital = max(1, current_capital)
    scale_labor = max(1, current_labor)
    scale_price = max(1, current_price)
    scale_demand = max(1, expected_demand[0])

    # Variables with scaling and lower bounds
    model.labor = pyo.Var(domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_labor), bounds=(1e-6, None))
    model.capital = pyo.Var(domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_capital), bounds=(1e-6, None))
    model.production = pyo.Var(domain=pyo.NonNegativeReals, initialize=1e-6)
    model.sales = pyo.Var(domain=pyo.NonNegativeReals, initialize=1e-6)

    # Objective
    def objective_rule(model):
        return (expected_price[0] * model.sales / scale_price
                - wage * model.labor / scale_labor
                - (model.capital - current_capital) * capital_price / scale_capital
                - depreciation_rate * expected_price[0] * current_inventory / scale_price)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def production_constraint_rule(model):
        return model.production == current_productivity * (model.capital ** capital_elasticity) * (model.labor ** (1 - capital_elasticity))
    model.production_constraint = pyo.Constraint(rule=production_constraint_rule)

    def budget_constraint_rule(model):
        return (wage * model.labor + (model.capital - current_capital) * capital_price <= budget * 1.0000001)
    model.budget_constraint = pyo.Constraint(rule=budget_constraint_rule)

    def sales_constraint_demand_rule(model):
        return model.sales <= expected_demand[0]
    model.sales_constraint_demand = pyo.Constraint(rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model):
        return model.sales <= current_inventory + model.production
    model.sales_constraint_inventory = pyo.Constraint(rule=sales_constraint_inventory_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        total_cost = pyo.value(wage * model.labor + (model.capital - current_capital) * capital_price)
        if total_cost > budget * 1.0000001:
            print(f"WARNING: Budget constraint violated. Total cost: {total_cost}, Budget: {budget}")

    # Check if the solver found an optimal solution
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        return {
            'optimal_labor': pyo.value(model.labor),
            'optimal_capital': pyo.value(model.capital),
            'optimal_production': pyo.value(model.production),
            'optimal_price': expected_price[0],
            'optimal_sales': [pyo.value(model.sales)],
            'optimal_inventory': [current_inventory + pyo.value(model.production) - pyo.value(model.sales)]
        }
    else:
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        return None
