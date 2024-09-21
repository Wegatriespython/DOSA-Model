using JuMP
using Ipopt
using LinearAlgebra
using MathOptInterface
MOI = MathOptInterface

include("Cost_minimisation.jl")


function round_results(results::Dict, digits::Int=4)
  rounded_results = Dict()
  for (key, value) in results
    rounded_results[key] = round.(value, digits=digits)
  end
  return rounded_results
end

function profit_maximization(
  current_capital,
  current_labor,
  current_price,
  current_productivity,
  expected_demand,
  expected_price,
  capital_price,
  capital_elasticity,
  current_inventory,
  depreciation_rate,
  expected_periods,
  discount_rate,
  budget,
  wage,
  capital_supply,
  labor_supply,
  current_debt,
  current_carbon_intensity,
  new_carbon_intensity,
  carbon_tax_rate,
  holding_costs)

  # Parameters

  A = current_productivity
  alpha = capital_elasticity
  delta = depreciation_rate
  rho = discount_rate
  w = wage
  c = capital_price
  K0 = current_capital
  S0 = current_inventory
  L0 = current_labor
  D0 = current_debt
  CI0 = current_carbon_intensity
  T = expected_periods
  N = length(expected_demand)
  dt = T / N


  # Time points
  t = range(0, stop=T, length=N + 1)

  # Expected demand and price functions
  D_dem = expected_demand
  P = expected_price

  # Discount factors
  discount_factors = [exp(-rho * t_i) for t_i in t[1:N]]

  # Create the optimization model
  model = Model(Ipopt.Optimizer)
  set_optimizer_attribute(model, "max_iter", 10000)
  set_optimizer_attribute(model, "tol", 1e-5)
  set_optimizer_attribute(model, "linear_solver", "ma57")
  set_optimizer_attribute(model, "mu_strategy", "adaptive")
  set_optimizer_attribute(model, "print_level", 0)



  # Decision variables
  @variable(model, K[1:N+1] >= 0)   # Capital stock
  @variable(model, S[1:N+1] >= 0)   # Inventory
  @variable(model, I[1:N] >= 0)     # Investment
  @variable(model, L[1:N] >= 0)     # Labor employed
  @variable(model, Y[1:N] >= 0)     # Sales
  @variable(model, Q[1:N] >= 0)     # Production

  @variable(model, Cash[1:N] >= 0)
  # Debt variables
  @variable(model, Debt[1:N+1] >= 0)         # Debt level
  @variable(model, NB[1:N] >= 0)             # Net borrowing
  @variable(model, IP[1:N] >= 0)             # Interest payment
  @variable(model, DP[1:N] >= 0)             # Debt payment

  # Carbon variables
  @variable(model, CI[1:N+1] >= 0)           # Carbon intensity
  @variable(model, E[1:N] >= 0)              # Emissions
  @variable(model, CTP[1:N] >= 0)            # Carbon tax payment

  max_capital = K0 + capital_supply
  max_labor = L0 + labor_supply/16


  set_start_value(K[1], (K0 + max_capital)/2)
  set_start_value(L[1], (L0+ max_labor)/2)
  # Initial conditions
  @constraint(model, S[1] == S0)
  @constraint(model, Debt[1] == D0)
  @constraint(model, CI[1] == CI0)
  @constraint(model, Cash[1] == budget)

  for i in 1:N
    @constraint(model, L[i] <= max_labor)
  end

  for i in 1:N
    @constraint(model, K[i] <= max_capital)
  end

  #Cash Constraints
  for i in 1:N
    if i == 1
      @constraint(model, Cash[i+1]  == NB[i] +Cash[1] + P[i] * Y[i] - w * L[i] - c * I[i] - DP[i]- IP[i] - CTP[i] - holding_costs * S[i])
    else
      @constraint(model, Cash[i] == Cash[i-1] + NB[i] + P[i] * Y[i] - w * L[i] - c * I[i] - DP[i] -IP[i]- CTP[i] - holding_costs * S[i])
    end
  end
  #Debug


  # Capital dynamicsj
  for i in 1:N
    if i == 1
      @constraint(model, K[i] == K0 + dt * (-delta * K0 + I[i]))
    else
      @constraint(model, K[i] == K[i-1] + dt * (-delta * K[i-1] + I[i]))
    end
  end

  # Inventory dynamics
  for i in 1:N
    if i == 1
      @constraint(model, S[i] == S0 + dt * (Q[1] - Y[1]))
    else
      @constraint(model, S[i] == S[i-1] + dt * (Q[i] - Y[i]))
    end
  end

  # Production function
  for i in 1:N
    @constraint(model, Q[i] == A * K[i]^alpha * L[i]^(1 - alpha))
  end

  # Demand constraint
  for i in 1:N
    @constraint(model, Y[i] <= D_dem[i])
  end

  # Inventory constraint
  for i in 1:N
    @constraint(model, Y[i] <= S[i])
  end

  # Emissions calculation
  for i in 1:N
    @constraint(model, E[i] == CI[i] * Q[i])
  end

  # Carbon tax payment
  for i in 1:N
    @constraint(model, CTP[i] == carbon_tax_rate * E[i])
  end

  # Carbon intensity evolution
  for i in 1:N
    if i == 1
      @constraint(model, CI[i+1] == ((1 - delta * dt) * CI[i] * K[i] + I[i] * new_carbon_intensity * dt) / K[i+1])
    else
      @constraint(model, CI[i+1] == ((1 - delta * dt) * CI[i] * K[i] + I[i] * new_carbon_intensity * dt) / K[i+1])
    end
  end

  # Debt dynamics
  interest_rate = rho  # Assuming discount rate as interest rate
  for i in 1:N
    @constraint(model, IP[i] == Debt[i] * interest_rate * dt)
    @constraint(model, Debt[i+1] == Debt[i] + NB[i] * dt - (DP[i] - IP[i]))
  end

  # Productive borrowing constraint
  for i in 1:N
    @constraint(model, NB[i] <= I[i])
  end

  # Terminal debt condition
  @constraint(model, Debt[N+1] <= D0)

  #Terminal inventory condtition
  @constraint(model, S[N+1] <= S0)


  # Objective function
  @objective(model, Max, sum(
    dt * discount_factors[i] * (
      P[i] * Y[i]
      -
      w * L[i]
      -
      c * I[i]
      -
      IP[i]
      -
      CTP[i]
      -
      holding_costs * S[i]
    ) for i in 1:N
  ))

  # Solve the optimization problem
  optimize!(model)

  # Check the solver status
  status = termination_status(model)
  if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    println("Optimal solution found.")
  else
    println("Solver did not find an optimal solution.")
  end

  unrounded_results = Dict(
    "optimal_labor" => value.(L),
    "optimal_capital" => value.(K),
    "optimal_production" => value.(Q),
    "optimal_investment" => value.(I),
    "optimal_sales" => value.(Y),
    "optimal_inventory" => value.(S),
    "optimal_debt" => value.(Debt),
    "optimal_net_borrowing" => value.(NB),
    "optimal_interest_payment" => value.(IP),
    "optimal_debt_payment" => value.(DP),
    "optimal_carbon_intensity" => value.(CI),
    "optimal_emissions" => value.(E),
    "optimal_carbon_tax_payment" => value.(CTP)
  )
  params = Dict(
    "current_productivity" => current_productivity,
    "capital_elasticity" => capital_elasticity,
    "depreciation_rate" => depreciation_rate,
    "discount_rate" => discount_rate,
    "wage" => wage,
    "capital_price" => capital_price,
    "current_capital" => current_capital,
    "current_inventory" => current_inventory,
    "current_debt" => current_debt,
    "current_carbon_intensity" => current_carbon_intensity,
    "expected_periods" => expected_periods,
    "expected_demand" => expected_demand,
    "expected_price" => expected_price,
    "discount_factors" => discount_factors,
    "dt" => dt
  )
  final = round_results(unrounded_results)
  #println(final)
  zero_profit_conditions = calculate_zero_profit_conditions(final, params)
  # Return the results
  return final, round_results(zero_profit_conditions)
end
