using JuMP
using Ipopt

function profit_maximization(
    current_capital, current_labor, current_price, current_productivity,
    expected_demand, expected_price, capital_price, capital_elasticity,
    current_inventory, depreciation_rate, expected_periods, discount_rate,
    budget, wage, capital_supply, labor_supply)

    max_labor = budget / wage
    max_capital = current_capital


    guess_capital = (current_capital + max_capital) / 2
    guess_labor = (current_labor + max_labor) / 2

    scale_capital = max(1, guess_capital)
    scale_labor = max(1, guess_labor)
    scale_price = max(1, maximum(expected_price))
    scale_demand = max(1, maximum(expected_demand))

    model = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", 10000)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "linear_solver", "ma57")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "print_level", 0)



    # Variables
    @variable(model, labor >= 1e-6, start = max(1e-6,guess_labor))
    @variable(model, capital >= 1e-6, start = max(1e-6,guess_capital))
    @variable(model, production >= 1e-6)
    @variable(model, inventory[1:expected_periods] >= 0)
    @variable(model, sales[1:expected_periods] >= 0)

    # Objective
    @objective(model, Max, sum(
        (expected_price[t] * sales[t] / scale_price
         - wage * labor / scale_labor
         - (capital - current_capital) * capital_price / scale_capital * (t == 1 ? 1 : 0)
         - depreciation_rate * expected_price[t] * inventory[t] / scale_price
        ) / ((1 + discount_rate) ^ (t-1))
        for t in 1:expected_periods
    ))

    # Constraints
    @constraint(model, production == current_productivity * (capital ^ capital_elasticity) * (labor ^ (1 - capital_elasticity)))
    @constraint(model, wage * labor + (capital - current_capital) * capital_price <= budget * 1.0000001)
    @constraint(model, inventory[1] == current_inventory + production - sales[1])
    @constraint(model, [t=2:expected_periods], inventory[t] == inventory[t-1] + production - sales[t])
    @constraint(model, [t=1:expected_periods], sales[t] <= expected_demand[t])
    @constraint(model, [t=1:expected_periods], sales[t] <= inventory[t] + production)
    # Solve
    optimize!(model)
    status = termination_status(model)
    println("Termination status: ", status)

      if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        return Dict(
            "optimal_labor" => value(labor),
            "optimal_capital" => value(capital),
            "optimal_production" => value(production),
            "optimal_price" => expected_price[1],
            "optimal_sales" => [value(sales[t]) for t in 1:expected_periods],
            "optimal_inventory" => [value(inventory[t]) for t in 1:expected_periods]
        )
    else
      println("Optimization failed.")
      println("Solver status: ", raw_status(model))
      println("Objective value: ", objective_value(model))
      println("Primal status: ", primal_status(model))
      println("Dual status: ", dual_status(model))
      return nothing

    end
end
