import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from functools import lru_cache

# Global variable to store the last solution for warm start
last_solution = None

@lru_cache(maxsize=2056)
def memoized_maximize_utility(savings, wages, prices, discount_rate, periods, alpha, max_working_hours, linear_solver):
    return _maximize_utility(savings, wages, prices, discount_rate, periods, alpha, max_working_hours, linear_solver)

def maximize_utility(Utility_params, linear_solver='ma57'):
    global last_solution
    savings = Utility_params['savings']
    wages = Utility_params['wage']
    prices = Utility_params['price']
    discount_rate = Utility_params['discount_rate']
    periods = Utility_params['time_horizon']
    alpha = Utility_params['alpha']
    max_working_hours = Utility_params['max_working_hours']
    result = memoized_maximize_utility(savings, tuple(wages), tuple(prices), discount_rate, periods, alpha, max_working_hours, linear_solver)

    if result is not None:
        last_solution = result  # Update last_solution for warm start
        rounded_result = round_to_integers(result)
        return rounded_result

    return result

def _maximize_utility(savings, wages, prices, discount_rate, periods, alpha, max_working_hours, linear_solver):
    global last_solution

    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, periods-1)

    # Variables
    model.consumption = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(1e-6, None))
    model.working_hours = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(1e-6, max_working_hours))
    model.leisure = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(1e-6
      , max_working_hours))
    model.savings = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # Parameters
    model.initial_savings = pyo.Param(initialize=savings)
    model.wages = pyo.Param(model.T, initialize=dict(enumerate(wages)))
    model.prices = pyo.Param(model.T, initialize=dict(enumerate(prices)))
    model.discount_rate = pyo.Param(initialize=discount_rate)
    model.alpha = pyo.Param(initialize=alpha)
    model.max_working_hours = pyo.Param(initialize=max_working_hours)
    min_consumtion = 0.5
    decay_rate = 0.9
    # Constraints
    @model.Constraint(model.T)
    def time_constraint(model, t):
        return model.leisure[t] + model.working_hours[t] == model.max_working_hours


    @model.Constraint(model.T)
    def savings_evolution(model,t):
        if t == 0:
            return model.savings[t] == model.initial_savings + model.wages[t] * model.working_hours[t] - model.prices[t] * model.consumption[t]
        else:
            return model.savings[t] == model.savings[t-1] + model.wages[t] * model.working_hours[t] - model.prices[t] * model.consumption[t]

    @model.Constraint(model.T)
    def consumption_constraint(model, t):
      return model.consumption[t] >= min_consumtion * decay_rate**t


    @model.Constraint()
    def terminal_savings_constraint(model):
        return model.savings[periods-1] >= 0

    # Objective
    @model.Objective(sense=pyo.maximize)
    def objective_rule(model):
        epsilon = 1e-6
        utility = sum((
            (((model.consumption[t] + epsilon)** (model.alpha)) *
             (model.leisure[t] + epsilon) ** (1 - model.alpha))) * (1/(1+model.discount_rate)**t)
            for t in model.T
        )
        return utility


    # Solve the model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 10000
    solver.options['tol'] = 1e-3
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = linear_solver

    solver.options['warm_start_init_point'] = 'yes'  # Use warm start


    solver.options['mu_strategy'] = 'adaptive'
    solver.options['print_level'] = 5  # Increase print level for more information


    solver.options['linear_scaling_on_demand'] = 'yes'  # Perform linear scaling only when needed




    try:
        results = solver.solve(model, tee=False)

        # Check if the solver found an optimal solution
        if (results.solver.status == pyo.SolverStatus.ok and
            results.solver.termination_condition == pyo.TerminationCondition.optimal):
            optimal_consumption = [pyo.value(model.consumption[t]) for t in model.T]
            optimal_working_hours = [pyo.value(model.working_hours[t]) for t in model.T]
            optimal_leisure = [pyo.value(model.leisure[t]) for t in model.T]
            optimal_savings = [pyo.value(model.savings[t]) for t in model.T]

            return optimal_consumption, optimal_working_hours, optimal_leisure, optimal_savings
        else:

            print(f"Solver status: {results.solver.status}")
            print(f"Termination condition: {results.solver.termination_condition}")
            return ([0] * periods, [13] * periods,
                    [0] * periods, [0] * periods)  # Default values if optimization fails
    except Exception as e:

        print(f"An error occurred during optimization: {str(e)}")
        return ([1] * periods, [13] * periods,
                [0] * periods, [0] * periods)  # Default values if optimization fails
def round_to_integers(solution):
    return tuple(
        [list(map(round, component)) for component in solution]
    )
