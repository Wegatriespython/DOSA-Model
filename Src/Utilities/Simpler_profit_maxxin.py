import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from functools import lru_cache

# Global variable to store the last solution for warm start
last_solution = None

@lru_cache(maxsize=2056)
def memoized_profit_maximization(
    current_capital, current_labor, current_price, current_productivity,
    expected_demand, expected_price, capital_price, capital_elasticity,
    current_inventory, depreciation_rate, expected_periods, discount_rate,
    budget, wage,capital_supply,labor_supply,current_debt, current_carbon_intensity,new_capital_carbon_intensity,carbon_tax_rate, linear_solver):
    return _profit_maximization(
      current_capital, current_labor, current_price, current_productivity,
              expected_demand, expected_price, capital_price, capital_elasticity,
              current_inventory, depreciation_rate, expected_periods, discount_rate,
              budget, wage,capital_supply, labor_supply, current_debt, current_carbon_intensity,new_capital_carbon_intensity,carbon_tax_rate, linear_solver)

def profit_maximization(
    current_capital, current_labor, current_price, current_productivity,
    expected_demand, expected_price, capital_price, capital_elasticity,
    current_inventory, depreciation_rate, expected_periods, discount_rate,
    budget, wage,capital_supply,labor_supply, current_debt, current_carbon_intensity, new_capital_carbon_intensity,carbon_tax_rate,  linear_solver='ma57'):

    global last_solution

    # Convert numpy arrays to tuples for hashing
    expected_demand_tuple = tuple(expected_demand)
    expected_price_tuple = tuple(expected_price)

    result = memoized_profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand_tuple, expected_price_tuple, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage,capital_supply,labor_supply, current_debt,current_carbon_intensity,new_capital_carbon_intensity,carbon_tax_rate, linear_solver)

    if result is not None:
        last_solution = (result['optimal_labor'], result['optimal_capital'])
        rounded_result = round_results(result)
        return rounded_result

    return result

def _profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage,capital_supply, labor_supply, current_debt, current_carbon_intensity, new_capital_carbon_intensity, carbon_tax_rate, linear_solver):

    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, expected_periods - 1)
    max_labor = labor_supply/16 + current_labor
    max_capital = capital_supply + current_capital


    guess_capital = (current_capital + max_capital)/2
    guess_labor = (current_labor + max_labor)/2

    interest_rate = 1/discount_rate - 1

    # Scaling factors
    scale_capital = max(1, guess_capital)
    scale_labor = max(1, guess_labor)
    scale_price = max(1, max(expected_price))
    scale_demand = max(1, max(expected_demand))

    # Variables with scaling and lower bounds
    model.labor = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_labor), bounds=(1e-6, max_labor))
    model.capital = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, guess_capital), bounds=(1e-6, max_capital))
    model.production = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, expected_demand[0]))
    model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, current_inventory))
    model.sales = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, expected_demand[0]))
    model.debt = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, current_capital))
    model.debt_payment = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=max(1e-6, current_debt * interest_rate))
    model.cash = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=budget)
    model.net_borrowing = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)
    model.investment = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.carbon_intensity = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=current_carbon_intensity)
    model.emissions = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)
    model.carbon_tax_payment = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)


    # Objective
    def objective_rule(model):
        obj_value = sum(
            (expected_price[t] * model.sales[t] / scale_price
             - wage * model.labor[t] / scale_labor
             - model.debt_payment[t]
             - depreciation_rate * expected_price[t] * model.inventory[t] / scale_price
             - model.carbon_tax_payment[t] / scale_price  # Use scale_price instead of scale_demand
            ) / ((1 + discount_rate) ** t)
            for t in model.T
        )
        return obj_value
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def carbon_tax_payment_rule(model, t):
        return model.carbon_tax_payment[t] == model.emissions[t] * carbon_tax_rate
    model.carbon_tax_payment_constraint = pyo.Constraint(model.T, rule=carbon_tax_payment_rule)


    def production_constraint_rule(model, t):
        return model.production[t] == current_productivity * (model.capital[t] ** capital_elasticity) * (model.labor[t] ** (1 - capital_elasticity))
    model.production_constraint = pyo.Constraint(model.T, rule=production_constraint_rule)

    def emissions_constraint_rule(model, t):
        return model.emissions[t] == model.carbon_intensity[t] * model.production[t]
    model.emissions_constraint = pyo.Constraint(model.T, rule=emissions_constraint_rule)
    def carbon_intensity_evolution_rule(model, t):
        if t == 0:
            return model.carbon_intensity[t] == ((1 - depreciation_rate) * current_capital * current_carbon_intensity +
                                                 model.investment[t] * new_capital_carbon_intensity) / model.capital[t]
        else:
            return model.carbon_intensity[t] == ((1 - depreciation_rate) * model.capital[t-1] * model.carbon_intensity[t-1] +
                                                 model.investment[t] * new_capital_carbon_intensity) / model.capital[t]
    model.carbon_intensity_evolution = pyo.Constraint(model.T, rule=carbon_intensity_evolution_rule)

    def cash_flow_rule(model, t):
        cash_beginning = budget if t == 0 else model.cash[t-1]
        cash_available = cash_beginning + model.net_borrowing[t]
        capital_expenditure = model.investment[t] * capital_price
        expenses = (wage * model.labor[t] + model.debt_payment[t] +
                    capital_expenditure + model.carbon_tax_payment[t])
        return cash_available >= expenses
    model.cash_flow_constraint = pyo.Constraint(model.T, rule=cash_flow_rule)

    def cash_balance_rule(model, t):
        # Beginning cash
        cash_beginning = budget if t == 0 else model.cash[t-1]
        # Expenses in t
        capital_expenditure = model.investment[t] * capital_price
        expenses = wage * model.labor[t] + model.debt_payment[t] + capital_expenditure + model.carbon_tax_payment[t]
        # Revenue in t
        revenue = model.sales[t] * expected_price[t]
        # Cash balance at end of t
        return model.cash[t] == cash_beginning + model.net_borrowing[t] - expenses + revenue
    model.cash_balance_constraint = pyo.Constraint(model.T, rule=cash_balance_rule)

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
        return model.sales[t] <= expected_demand[t]
    model.sales_constraint_demand = pyo.Constraint(model.T, rule=sales_constraint_demand_rule)

    def sales_constraint_inventory_rule(model, t):
        if t == 0:
            available_inventory = current_inventory + model.production[t]
        else:
            available_inventory = model.inventory[t-1] + model.production[t]
        return model.sales[t] <= available_inventory
    model.sales_constraint_inventory = pyo.Constraint(model.T, rule=sales_constraint_inventory_rule)


    def debt_evolution_rule(model, t):
        previous_debt = current_debt if t == 0 else model.debt[t-1]
        return model.debt[t] == previous_debt * (1 + interest_rate) + model.net_borrowing[t] - model.debt_payment[t]
    model.debt_evolution_constraint = pyo.Constraint(model.T, rule=debt_evolution_rule)


    def debt_payment_rule(model, t):
        previous_debt = current_debt if t == 0 else model.debt[t-1]
        interest_amount = previous_debt * interest_rate
        return model.debt_payment[t] >= interest_amount
    model.debt_payment_constraint = pyo.Constraint(model.T, rule=debt_payment_rule)


    def terminal_debt_rule(model):
        return model.debt[model.T.last()] == 0
    model.terminal_debt_constraint = pyo.Constraint(rule=terminal_debt_rule)




    def borrowing_limit_rule(model, t):
        return model.debt[t] <= 20
    model.borrowing_limit_constraint = pyo.Constraint(model.T, rule=borrowing_limit_rule)



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


   # Check if the solver found an optimal solution
    if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):

        profits_per_period = []
        for t in model.T:
            profit = (
                expected_price[t] * pyo.value(model.sales[t]) / scale_price
                - wage * pyo.value(model.labor[t]) / scale_labor
                - pyo.value(model.debt_payment[t])
                - depreciation_rate * expected_price[t] * pyo.value(model.inventory[t]) / scale_price
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
            'optimal_carbon_tax_payments': [pyo.value(model.carbon_tax_payment[t]) for t in model.T]
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
            rounded[key] = [round(v,2) for v in value]
        else:
            rounded[key] = round(value, 2)
    return rounded
