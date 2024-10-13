import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import logging
from functools import lru_cache
from Utilities.Partial_EQ_cost_min import cost_minimization
from pyomo.util.infeasible import log_infeasible_constraints
# Global variable to store the last solution for warm start
last_solution = None



def profit_maximization(Profit_max_params, linear_solver='ma57'):

    global last_solution

    result = _profit_maximization(Profit_max_params, linear_solver)

    if result is not None:
        last_solution = (result['optimal_labor'], result['optimal_capital'])
        rounded_result = round_results(result)

                # Perform cost minimization
        params = {
            'current_price': Profit_max_params['current_price'],
            'wage': Profit_max_params['wage'],
            'capital_price': Profit_max_params['capital_price'],
            'depreciation_rate': Profit_max_params['depreciation_rate'],
            'holding_costs': Profit_max_params['holding_costs'],
            'carbon_tax_rate': Profit_max_params['carbon_tax_rate'],
            'periods': Profit_max_params['time_horizon'],
            'inventory': Profit_max_params['inventory'],
        }

        """zero_profit_result, check = cost_minimization(rounded_result, params)
        if check != False:
          zero_profit_result = round_results(zero_profit_result, 2)
        else:
            zero_profit_result = None"""


        return rounded_result, None

    return None

def _profit_maximization(Profit_max_params, linear_solver):

  # Unpack parameters
    current_capital = Profit_max_params['current_capital']
    current_labor = Profit_max_params['current_labor']
    current_price = Profit_max_params['current_price']
    current_productivity = Profit_max_params['productivity']
    expected_demand = Profit_max_params['expected_demand']
    expected_price = Profit_max_params['expected_price']
    capital_price = Profit_max_params['capital_price']
    capital_elasticity = Profit_max_params['capital_elasticity']
    current_inventory = Profit_max_params['inventory']
    depreciation_rate = Profit_max_params['depreciation_rate']
    expected_periods = Profit_max_params['time_horizon']
    discount_rate = Profit_max_params['discount_rate']
    budget = Profit_max_params['budget']
    wage = Profit_max_params['wage']
    capital_supply = Profit_max_params['expected_capital_supply']
    labor_supply = Profit_max_params['expected_labor_supply']
    holding_costs = Profit_max_params['holding_costs']
    carbon_tax_rate = Profit_max_params['carbon_tax_rate']
    new_capital_carbon_intensity = Profit_max_params['new_capital_carbon_intensity']
    current_carbon_intensity = Profit_max_params['carbon_intensity']
    current_debt = Profit_max_params['debt']


    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, expected_periods - 1)
    max_labor = max(labor_supply/16 + current_labor +1e-6, 30/5)
    max_capital = current_capital + 1e-6
    #(capital_supply if capital_supply > 0 else current_capital) + current_capital
    #double current capital when capital supllies are not available


    guess_capital = (current_capital + max_capital)/2
    guess_labor = (current_labor + max_labor)/2

    interest_rate = discount_rate
    print(interest_rate)

    # Scaling factors
    scale_capital = max(1, guess_capital)
    scale_labor = max(1, guess_labor)
    scale_price = max(1, max(expected_price))
    scale_demand = max(1, max(expected_demand))

    # Variables with scaling and lower bounds
    model.labor = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(0, guess_labor), bounds=(0, max_labor))
    model.capital = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(0, current_capital), bounds=(0, max_capital))
    model.production = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(0, expected_demand[0]))
    model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(0, current_inventory))
    model.sales = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(0, expected_demand[0]))
    model.cash = pyo.Var(model.T, domain=pyo.Reals, initialize=max(0, budget))
    model.investment = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize = max(1e-6, current_capital), bounds =(0,max_capital))
    model.carbon_intensity = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(0, current_carbon_intensity))
    model.emissions = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)
   # Real Constraints
    model.carbon_tax_payment = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)
    model.net_borrowing = pyo.Var(model.T, domain = pyo.NonNegativeReals, initialize =0)
    model.interest_payment = pyo.Var(model.T, domain= pyo.NonNegativeReals, initialize =0)
    model.debt = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=current_debt)
    model.debt_payment = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)

    # Objective
    def objective_rule(model, t):

      obj_value = sum((
            (expected_price[t] * model.sales[t])/scale_price -
            (wage * model.labor[t]/scale_labor  +
              depreciation_rate * model.capital[t]/scale_capital +
              (capital_price * model.investment[t])/scale_capital + model.inventory[t]* holding_costs/scale_price + model.interest_payment[t]/scale_price + model.carbon_tax_payment[t]/scale_price)
            ) * (1/(1 + discount_rate))**t
            for t in model.T
        )
      return obj_value

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def carbon_tax_payment_rule(model, t):
        return model.carbon_tax_payment[t] == model.emissions[t] * carbon_tax_rate
    model.carbon_tax_payment_constraint = pyo.Constraint(model.T, rule=carbon_tax_payment_rule)

    def production_constraint_rule(model, t):
        episilon = 1e-6
        return model.production[t] == current_productivity * ((model.capital[t] + episilon) ** capital_elasticity) * ((model.labor[t]+ episilon)** (1 - capital_elasticity))
    model.production_constraint = pyo.Constraint(model.T, rule=production_constraint_rule)

    def emissions_constraint_rule(model, t):
        return model.emissions[t] == model.carbon_intensity[t] * model.production[t]
    model.emissions_constraint = pyo.Constraint(model.T, rule=emissions_constraint_rule)

    def carbon_intensity_evolution_rule(model, t):
        epsilon = 1e-6
        if t == 0:
            return model.carbon_intensity[t] == ((1 - depreciation_rate) * current_capital * current_carbon_intensity +
                                                 model.investment[t] * new_capital_carbon_intensity) / (model.capital[t] + epsilon)
        else:
            return model.carbon_intensity[t] == ((1 - depreciation_rate) * model.capital[t-1] * model.carbon_intensity[t-1] +
                                                 model.investment[t] * new_capital_carbon_intensity) / (model.capital[t] + epsilon)
    model.carbon_intensity_evolution = pyo.Constraint(model.T, rule=carbon_intensity_evolution_rule)

    def cash_balance_constraint(model, t):
        if t == 0:
            return model.cash[t] == budget + model.net_borrowing[t] + expected_price[t] * model.sales[t] - (
                wage * model.labor[t] + capital_price * model.investment[t] +
                holding_costs * current_inventory + model.debt_payment[t]+
                model.carbon_tax_payment[t])
        else:
            return model.cash[t] == model.cash[t-1] + model.net_borrowing[t] + expected_price[t] * model.sales[t] - (
                wage * model.labor[t] + capital_price * model.investment[t] +
                holding_costs * model.inventory[t-1] + model.debt_payment[t] +
                model.carbon_tax_payment[t])
    model.cash_balance_constraint = pyo.Constraint(model.T, rule=cash_balance_constraint)

    def productive_borrowing_constraint(model, t):
        return model.net_borrowing[t] <= model.investment[t] #Borrowing can only be used for investment
    model.productive_borrowing_constraint = pyo.Constraint(model.T, rule=productive_borrowing_constraint)


    def inventory_balance_rule(model, t):
        if t == 0:
            return model.inventory[t] == current_inventory + model.production[t] - model.sales[t]
        else:
            return model.inventory[t] == model.inventory[t-1] + model.production[t] - model.sales[t]
    model.inventory_balance = pyo.Constraint(model.T, rule=inventory_balance_rule)

    def capital_constraint_rule(model, t):
        if t == 0:
            return model.capital[t] == (1 - depreciation_rate) * current_capital + model.investment[t]
        else:
            return model.capital[t] == (1 - depreciation_rate) * model.capital[t-1] + model.investment[t]
    model.capital_constraint = pyo.Constraint(model.T, rule=capital_constraint_rule)


    def sales_constraint_demand_rule(model, t):
        return model.sales[t] <= expected_demand[t] * 1.0001
    model.sales_constraint_demand = pyo.Constraint(model.T, rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model, t):
        if t == 0:
            available_inventory = current_inventory + model.production[t]
        else:
            available_inventory = model.inventory[t-1] + model.production[t]
        return model.sales[t] <= available_inventory + 1e-6
    model.sales_constraint_inventory = pyo.Constraint(model.T, rule=sales_constraint_inventory_rule)

    def debt_payment_rule(model, t):
        previous_debt = current_debt if t == 0 else model.debt[t-1]
        model.interest_payment[t] = previous_debt * interest_rate
        # Ensure debt_payment[t] >= interest_payment[t]
        return model.debt_payment[t] >= model.interest_payment[t]
    model.debt_payment_constraint = pyo.Constraint(model.T, rule=debt_payment_rule)


    def debt_evolution_rule(model, t):
        if t == 0:
            return model.debt[t] <= current_debt + model.net_borrowing[t] - (model.debt_payment[t] - model.interest_payment[t])
        else:
            return model.debt[t] <= model.debt[t-1] + model.net_borrowing[t] - (model.debt_payment[t] - model.interest_payment[t])
    model.debt_evolution_constraint = pyo.Constraint(model.T, rule=debt_evolution_rule)


    def terminal_debt_rule(model,t):
        return model.debt[model.T.last()] <= current_debt
    model.terminal_debt_constraint = pyo.Constraint(rule=terminal_debt_rule)

    def terminal_inventory_rule(model,t):
        return model.inventory[model.T.last()] <= current_inventory
    model.terminal_inventory_constraint = pyo.Constraint(rule=terminal_inventory_rule)

    def borrowing_limit_rule(model, t):
        return model.net_borrowing[t] <= 10  #temporary disabling debt till model behavior is analysed wo debt
    model.borrowing_limit_constraint = pyo.Constraint(model.T, rule=borrowing_limit_rule)

    def labor_limit_rule(model, t):
        return model.labor[t] <= max_labor
    model.labor_limit_constraint = pyo.Constraint(model.T, rule=labor_limit_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-5  # Sets the convergence tolerance for the optimization algorithm
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver

    solver.options['warm_start_init_point'] = 'yes'  # Use warm start


    solver.options['mu_strategy'] = 'adaptive'
    solver.options['print_level'] = 5  # Increase print level for more information


    solver.options['linear_scaling_on_demand'] = 'yes'  # Perform linear scaling only when needed

    results = solver.solve(model, tee=False)


   # Check if the solver found an optimal solution
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):

        profits_per_period = []
        for t in model.T:
            profit = (
                expected_price[t] * pyo.value(model.sales[t])
                - wage * pyo.value(model.labor[t])
                - pyo.value(model.debt_payment[t])
                - capital_price * pyo.value(model.investment[t])
                - holding_costs * pyo.value(model.inventory[t-1] if t > 0 else 0)
                - pyo.value(model.carbon_tax_payment[t])

            )
            profits_per_period.append(profit)


        unrounded_results = {
            'optimal_labor': [pyo.value(model.labor[t]) for t in model.T],
            'optimal_capital': [pyo.value(model.capital[t]) for t in model.T],
            'optimal_production': [pyo.value(model.production[t]) for t in model.T],
            'optimal_price': expected_price[0],
            'optimal_investment': [pyo.value(model.investment[t]) for t in model.T],
            'profits_per_period': profits_per_period,
            'objective': pyo.value(model.objective),
            'optimal_sales': [pyo.value(model.sales[t]) for t in model.T],
            'optimal_inventory': [pyo.value(model.inventory[t]) for t in model.T],
            'optimal_debt': [pyo.value(model.debt[t]) for t in model.T],
            'optimal_debt_payment': [pyo.value(model.debt_payment[t]) for t in model.T],
            'optimal_carbon_intensity': [pyo.value(model.carbon_intensity[t]) for t in model.T],
            'optimal_emissions': [pyo.value(model.emissions[t]) for t in model.T],
            'optimal_carbon_tax_payment': [pyo.value(model.carbon_tax_payment[t]) for t in model.T],
            'optimal_net_borrowing': [pyo.value(model.net_borrowing[t]) for t in model.T],
            'optimal_interest_payment': [pyo.value(model.interest_payment[t]) for t in model.T]
        }
        return round_results(unrounded_results)
    else:
        breakpoint()
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        return None
def round_results(results, precision=0):
    rounded = {}
    for key, value in results.items():
        if isinstance(value, list):
            rounded[key] = [round(v, precision) for v in value]
        else:
            rounded[key] = round(value, precision)
    return rounded
def log_pyomo_infeasible_constraints(model_instance):
    # Create a logger object with DEBUG level
    logging_logger = logging.getLogger()
    logging_logger.setLevel(logging.DEBUG)
    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add the handler to the logger
    logging_logger.addHandler(ch)
    # Log the infeasible constraints of pyomo object
    print("Displaying Infeasible Constraints")
    log_infeasible_constraints(model_instance, log_expression=True,
                          log_variables=True, logger=logging_logger)
