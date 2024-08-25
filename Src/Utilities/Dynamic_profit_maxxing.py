import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

def dynamic_profit_maximization(
        initial_capital, initial_labor, initial_price, productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        initial_inventory, depreciation_rate, periods, discount_rate,
        budget, wage, terminal_value_rate=0.5, linear_solver='ma27'):

    model = pyo.ConcreteModel()
    max_labor = budget/max(1e-6,wage) + initial_labor
    max_capital = budget/max(1e-6,capital_price) + initial_capital

    guess_capital = (initial_capital + max_capital)/2
    guess_labor = (initial_labor + max_labor)/2
    # Sets
    model.T = pyo.RangeSet(0, periods - 1)

    # Parameters
    model.productivity = pyo.Param(initialize=productivity)
    model.expected_demand = pyo.Param(model.T, initialize=lambda m, t: expected_demand[t])
    model.expected_price = pyo.Param(model.T, initialize=lambda m, t: expected_price[t])
    model.budget = pyo.Param(model.T, initialize=lambda m, t: budget[t] if isinstance(budget, list) else budget)
    model.wage = pyo.Param(initialize=wage)
    model.capital_price = pyo.Param(initialize=capital_price)
    model.depreciation_rate = pyo.Param(initialize=depreciation_rate)
    model.discount_rate = pyo.Param(initialize=discount_rate)
    model.capital_elasticity = pyo.Param(initialize=capital_elasticity)

    # Variables with small lower bounds
    model.labor = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(guess_labor, 1e-6), bounds=(1e-6, None))
    model.capital = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(guess_capital, 1e-6), bounds=(1e-6, None))
    model.investment = pyo.Var(model.T, domain=pyo.Reals, initialize=0)
    model.production = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1e-6, bounds=(1e-6, None))
    model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(initial_inventory, 1e-6), bounds=(1e-6, None))
    model.sales = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1e-6, bounds=(1e-6, None))

    # Objective
    def objective_rule(model):
        return sum(
            (model.expected_price[t] * model.sales[t]
             - model.wage * model.labor[t]
             - model.capital_price * model.investment[t]
             - model.depreciation_rate * model.expected_price[t] * model.inventory[t]
            ) / ((1 + model.discount_rate) ** t)
            for t in model.T
        ) + (terminal_value_rate * model.capital[model.T.last()] * model.capital_price) / ((1 + model.discount_rate) ** model.T.last())
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def production_constraint_rule(model, t):
        return model.production[t] == model.productivity * (model.capital[t] ** model.capital_elasticity) * (model.labor[t] ** (1 - model.capital_elasticity))
    model.production_constraint = pyo.Constraint(model.T, rule=production_constraint_rule)

    def budget_constraint_rule(model, t):
        return model.wage * model.labor[t] + model.capital_price * model.investment[t] <= model.budget[t]
    model.budget_constraint = pyo.Constraint(model.T, rule=budget_constraint_rule)

    def capital_accumulation_rule(model, t):
        if t == 0:
            return model.capital[t] == initial_capital + model.investment[t]
        else:
            return model.capital[t] == (1 - model.depreciation_rate) * model.capital[t-1] + model.investment[t]
    model.capital_accumulation = pyo.Constraint(model.T, rule=capital_accumulation_rule)

    def inventory_balance_rule(model, t):
        if t == 0:
            return model.inventory[t] == initial_inventory + model.production[t] - model.sales[t]
        else:
            return model.inventory[t] == model.inventory[t-1] + model.production[t] - model.sales[t]
    model.inventory_balance = pyo.Constraint(model.T, rule=inventory_balance_rule)

    def sales_constraint_demand_rule(model, t):
        return model.sales[t] <= model.expected_demand[t]
    model.sales_constraint_demand = pyo.Constraint(model.T, rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model, t):
        return model.sales[t] <= model.inventory[t] + model.production[t]
    model.sales_constraint_inventory = pyo.Constraint(model.T, rule=sales_constraint_inventory_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver

    try:
        results = solver.solve(model, tee=False)  # Set tee=True for more solver output

        # Check if the solver found an optimal solution
        if (results.solver.status == pyo.SolverStatus.ok and
            results.solver.termination_condition == pyo.TerminationCondition.optimal):
            return {
                'optimal_labor': pyo.value(model.labor[0]),
                'optimal_capital': pyo.value(model.capital[0]),
                'optimal_investment': pyo.value(model.investment[0]),
                'optimal_production': pyo.value(model.production[0]),
                'optimal_sales': pyo.value(model.sales[0]),
                'optimal_inventory': pyo.value(model.inventory[0]),
                'objective_value': pyo.value(model.objective)
            }
        else:
            print(f"Solver status: {results.solver.status}")
            print(f"Termination condition: {results.solver.termination_condition}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Model debugging information:")
        model.pprint()
        return None
