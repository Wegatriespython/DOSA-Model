import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def cost_minimization(profit_max_result, params):
    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, params['expected_periods'] - 1)

    # Variables
    model.price = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=params['current_price'],
      bounds=(params['current_price'], None)) # Upper bound can be adjusted
    model.wage = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=params['wage'],
      bounds= (params['wage'], None))
    model.capital_price = pyo.Var(model.T, domain = pyo.NonNegativeReals, initialize = params['capital_price'], bounds = (params['capital_price'], None))

    # Upper bound can be adjusted

    # Fixed variables from profit maximization
    model.production = profit_max_result['optimal_production']
    model.sales = profit_max_result['optimal_sales']
    model.capital = profit_max_result['optimal_capital']
    model.labor = profit_max_result['optimal_labor']
    model.inventory = profit_max_result['optimal_inventory']
    model.emissions = profit_max_result['optimal_emissions']

    # Objective: Minimize total costs
    def objective_rule(model):
        return sum(
            (model.wage[t] * model.labor[t] +
              model.capital_price[t] * params['depreciation_rate'] * model.capital[t] +
              params['holding_costs'] * model.inventory[t] +
              params['carbon_tax_rate'] * model.emissions[t])
            for t in model.T
        )
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraint: Zero profit condition
    def zero_profit_rule(model, t):
        revenue = model.price[t] * model.sales[t]
        costs = (model.wage[t] * model.labor[t] +
                  params['capital_price'] * params['depreciation_rate'] * model.capital[t] +
                  params['holding_costs'] * model.inventory[t] +
                  params['carbon_tax_rate'] * model.emissions[t])
        return revenue == costs
    model.zero_profit = pyo.Constraint(model.T, rule=zero_profit_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    results = solver.solve(model, tee=False)

    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        return {
            'price': [pyo.value(model.price[t]) for t in model.T],
            'wage': [pyo.value(model.wage[t]) for t in model.T],
            'capital_price': [pyo.value(model.capital_price[t]) for t in model.T]
        }
    else:
        print("Cost minimization solver failed to find an optimal solution.")
        return None
