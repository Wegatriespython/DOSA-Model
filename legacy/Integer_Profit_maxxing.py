import gurobipy as gp
from gurobipy import GRB

def profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage):

    try:
        # Create a new model
        model = gp.Model("ProfitMaximization")

        # Sets
        T = range(expected_periods)

        # Pre-calculate discount factors
        discount_factors = [(1 / (1 + discount_rate)) ** t for t in T]

        # Variables
        labor = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="labor")
        capital = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="capital")
        production = model.addVars(T, lb=0, name="production")
        inventory = model.addVars(T, lb=0, name="inventory")
        sales = model.addVars(T, lb=0, name="sales")

        # Additional variables for Cobb-Douglas function
        capital_part = model.addVars(T, lb=0, name="capital_part")
        labor_part = model.addVars(T, lb=0, name="labor_part")

        # Objective function
        obj = gp.QuadExpr()
        for t in T:
            revenue = expected_price[t] * sales[t]
            labor_cost = wage * labor[t]
            capital_cost = capital_price * (capital[t] - (1 - depreciation_rate) * capital[t-1] if t > 0 else capital[t] - current_capital)
            inventory_cost = depreciation_rate * expected_price[t] * inventory[t]
            profit = revenue - labor_cost - capital_cost - inventory_cost
            obj += profit * discount_factors[t]
        model.setObjective(obj, GRB.MAXIMIZE)

        # Production constraint using Cobb-Douglas function
        for t in T:
            model.addGenConstrPow(capital[t], capital_part[t], capital_elasticity, name=f"capital_part_{t}")
            model.addGenConstrPow(labor[t], labor_part[t], 1 - capital_elasticity, name=f"labor_part_{t}")
            model.addConstr(production[t] == current_productivity * capital_part[t] * labor_part[t], name=f"production_constraint_{t}")

        # Budget constraints
        for t in T:
            if t == 0:
                model.addConstr(wage * labor[t] + capital_price * (capital[t] - current_capital) <= budget, f"budget_constraint_{t}")
            else:
                model.addConstr(wage * labor[t] + capital_price * (capital[t] - (1 - depreciation_rate) * capital[t-1]) <= budget, f"budget_constraint_{t}")

        # Other constraints
        for t in T:
            if t == 0:
                model.addConstr(inventory[t] == current_inventory + production[t] - sales[t], f"inventory_balance_{t}")
            else:
                model.addConstr(inventory[t] == inventory[t-1] + production[t] - sales[t], f"inventory_balance_{t}")
            model.addConstr(sales[t] <= expected_demand[t], f"sales_constraint_demand_{t}")
            model.addConstr(sales[t] <= inventory[t] + production[t], f"sales_constraint_inventory_{t}")

        # Capital accumulation
        for t in range(1, expected_periods):
            model.addConstr(capital[t] >= (1 - depreciation_rate) * capital[t-1], f"capital_accumulation_{t}")

        # Set Gurobi parameters
        model.Params.NonConvex = 2  # Allow non-convex quadratic optimization
        model.Params.MIPGap = 0.01  # Set the MIP gap tolerance
        model.Params.TimeLimit = 6  # Set a time limit of 6 seconds
        model.Params.MIPFocus = 1  # Focus on finding feasible solutions quickly
        model.Params.Heuristics = 0.8  # Increase time spent on heuristics
        model.Params.Cuts = 2  # Aggressive cut generation

        # Optimize model
        model.optimize()

        # Check if the solver found an optimal solution
        if model.status == GRB.OPTIMAL:
            return {
                'optimal_labor': [labor[t].X for t in T],
                'optimal_capital': [capital[t].X for t in T],
                'optimal_production': [production[t].X for t in T],
                'optimal_price': expected_price[0],
                'optimal_sales': [sales[t].X for t in T],
                'optimal_inventory': [inventory[t].X for t in T],
                'optimal_profit': model.ObjVal
            }
        else:
            print(f"Optimization was stopped with status {model.status}")
            return None

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except AttributeError as e:
        print(f"Attribute error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
