import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from pyomo.util.infeasible import log_infeasible_constraints
"""
Big idea, cost minimisation needs to provide the zero profit conditions for the firm. Ideally firms operating with this as the lower margin are sustainable without failure. There is one issue however which is that firms are not likely to face their zero profit conditions in all three markets at once, wheras our model evaluates the three jointly.For example, firms could potentially be price setters in x =< 3 markets. In that case they need the zero profit conditions for only x not for all three. For example if the labor market is under oversupply then they can be wage setters and thus have far more wiggle room for consumtion prices and capital prices.

But its also possible that firms have heterogenous conditions where they don't uniformly have advantages.
One possible solution is to let the cost minimisation solve for scenarios, or one free variable at a time and then have an all three case. So the firm would have a zero-profit-dictionary for price references for each case it could possibly encounter.
Would that be exhaustive?

Match Scenario:
   Optimals  = x_o*L_o,y_o*Q_o,z_o*K_o
   Actuals   = x_a*L_a,y_a*Q_a,z_a*K_a

"""
def fake_result(params, profit_max_result):

  current_price = params['current_price']
  periods = params['periods']
  capital_price = params['capital_price']
  wage = params['wage']

  price = np.full(periods, current_price)
  wage = np.full(periods, wage)
  capital_price = np.full(periods, capital_price)
  result = {
    'price': price,
    'wage': wage,
    'capital_price': capital_price
  }
  return result
def cost_minimization(profit_max_result, params):

  sales = profit_max_result['optimal_sales']

  if np.mean(sales) != 0:
    return _cost_minimization(profit_max_result, params)
  else:
    result = fake_result(params, profit_max_result)
    return result, False

def _cost_minimization(profit_max_result, params):
    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, params['periods'] - 1)


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
        return revenue <= costs
    model.zero_profit = pyo.Constraint(model.T, rule=zero_profit_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    results = solver.solve(model, tee=False)
    log_infeasible_constraints(model)
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        return {
            'price': [pyo.value(model.price[t]) for t in model.T],
            'wage': [pyo.value(model.wage[t]) for t in model.T],
            'capital_price': [pyo.value(model.capital_price[t]) for t in model.T]
        } , True
    else:
        print("Cost minimization solver failed to find an optimal solution.")
        results = fake_result(params, profit_max_result)
        return results, False
