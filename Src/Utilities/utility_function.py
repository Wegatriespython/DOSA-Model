import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from functools import lru_cache

# Global variable to store the last solution for warm start
last_solution = None

@lru_cache(maxsize=128)
def memoized_maximize_utility(savings, current_wage, current_price, historical_price, working_hours, discount_factor, periods, alpha):
    return _maximize_utility(savings, current_wage, current_price, historical_price, working_hours, discount_factor, periods, alpha)

def maximize_utility(savings, current_wage, current_price, historical_price, working_hours, discount_factor=0.95, periods=2, alpha=0.9, linear_solver='ma27'):
    # This wrapper function handles the linear_solver parameter and warm start
    global last_solution

    result = memoized_maximize_utility(savings, current_wage, current_price, historical_price, working_hours, discount_factor, periods, alpha)

    if result is not None:
        optimal_consumption, optimal_working_hours, optimal_savings, optimal_future_working_hours = result
        last_solution = (optimal_consumption, optimal_working_hours, optimal_future_working_hours)  # Update last_solution for warm start

    return result

def _maximize_utility(savings, current_wage, current_price, historical_price, working_hours, discount_factor, periods, alpha, linear_solver='ma27'):
    global last_solution

    model = pyo.ConcreteModel()

    # Variables
    model.consumption = pyo.Var(domain=pyo.PositiveReals, initialize=1.0, bounds=(1, None))
    model.working_hours = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 16.001), initialize=16)
    model.leisure = pyo.Var(domain=pyo.PositiveReals, bounds=(1e-6, 16), initialize=0.000001)
    model.future_working_hours = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 16.001), initialize=16)

    # Parameters
    model.savings = pyo.Param(initialize=savings)
    model.current_wage = pyo.Param(initialize=current_wage)
    model.current_price = pyo.Param(initialize=current_price)
    model.historical_price = pyo.Param(initialize=historical_price)
    model.discount_factor = pyo.Param(initialize=discount_factor)
    model.periods = pyo.Param(initialize=periods)
    model.alpha = pyo.Param(initialize=alpha)

    # Auxiliary variables
    model.new_savings = pyo.Var(domain=pyo.Reals)
    model.future_consumption = pyo.Var(domain=pyo.PositiveReals, bounds=(1, None))
    model.future_income = pyo.Var(domain=pyo.Reals, bounds=(0, None))

    # Constraints
    @model.Constraint()
    def leisure_constraint(model):
        return model.leisure == 16 - model.working_hours

    @model.Constraint()
    def future_leisure_constraint(model):
        return 16 - model.future_working_hours >= 1e-6

    @model.Constraint()
    def new_savings_constraint(model):
        return (model.new_savings == model.savings + model.current_wage * model.working_hours
                - (model.consumption * model.current_price))

    @model.Constraint()
    def future_income_constraint(model):
        return model.future_income == model.current_wage * model.future_working_hours

    @model.Constraint()
    def future_consumption_constraint_lower(model):
        return model.future_consumption >= 1

    @model.Constraint()
    def future_consumption_constraint_upper(model):
        return model.future_consumption <= (model.new_savings + model.future_income) / (model.current_price * (model.periods - 1))

    @model.Constraint()
    def budget_constraint(model):
        minimum_savings = model.current_price * (model.periods - 1)  # Enough for 1 unit per future period
        return (model.savings + model.current_wage * model.working_hours
                >= model.consumption * model.current_price + minimum_savings)

    # Objective
    def objective_rule(model):
        epsilon = 1e-6  # Small constant to prevent zero values
        current_utility = pyo.log(model.consumption + epsilon) * model.alpha + pyo.log(model.leisure + epsilon) * (1-model.alpha)

        price_expectation = max(0, min(1, (model.current_price / model.historical_price - 1) * 0.5))
        price_adjusted_consumption = model.consumption * (1 - price_expectation) + epsilon

        future_leisure = 16 - model.future_working_hours + epsilon
        future_utility = sum(model.discount_factor**t * (pyo.log(model.future_consumption + epsilon) * model.alpha +
                                                         pyo.log(future_leisure) * (1-model.alpha))
                             for t in range(1, model.periods + 1))

        total_utility = current_utility + pyo.log(price_adjusted_consumption + epsilon) * model.alpha + pyo.log(model.leisure + epsilon) * (1-model.alpha) + future_utility
        return total_utility

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-6
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver
    # Warm start
    if last_solution is not None:
        model.consumption.value = last_solution[0]
        model.working_hours.value = min(last_solution[1], 16)
        model.future_working_hours.value = min(last_solution[2], 16)
    try:
        results = solver.solve(model, tee=False)

        # Check if the solver found an optimal solution
        if (results.solver.status == pyo.SolverStatus.ok and
            results.solver.termination_condition == pyo.TerminationCondition.optimal):
            optimal_consumption = pyo.value(model.consumption)
            optimal_working_hours = pyo.value(model.working_hours)
            optimal_savings = pyo.value(model.new_savings)
            optimal_future_working_hours = pyo.value(model.future_working_hours)

            return optimal_consumption, optimal_working_hours, optimal_savings, optimal_future_working_hours
        else:
            print(f"Solver status: {results.solver.status}")
            print(f"Termination condition: {results.solver.termination_condition}")
            return 0,16,0,16 # Work full time if the solver fails
    except Exception as e:
        print(f"An error occurred during optimization: {str(e)}")
        return 0,16,0,16
