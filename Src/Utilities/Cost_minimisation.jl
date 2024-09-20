using NLsolve
using ForwardDiff

function calculate_zero_profit_conditions(
    optimal_results::Dict,
    parameters::Dict
)
    # Extract optimal production decisions from results
    L_opt = optimal_results["optimal_labor"]
    I_opt = optimal_results["optimal_investment"]
    Y_opt = optimal_results["optimal_sales"]
    Q_opt = optimal_results["optimal_production"]
    CTP_opt = optimal_results["optimal_carbon_tax_payment"]
    IP_opt = optimal_results["optimal_interest_payment"]

    # Extract parameters
    dt = parameters["dt"]
    discount_factors = parameters["discount_factors"]
    w_fixed = parameters["wage"]
    c_fixed = parameters["capital_price"]
    P_fixed = parameters["expected_price"]

    # Define the total profit function as a function of the variable of interest
    N = length(Y_opt)  # Number of periods

    # Helper function to calculate profit given variable values
    function calculate_profit(variable_name::Symbol, variable_value::Float64)
        total_profit = 0.0
        for i in 1:N
            # Adjust the variable of interest
            if variable_name == :price
                P_i = variable_value
                w_i = w_fixed
                c_i = c_fixed
            elseif variable_name == :wage
                P_i = P_fixed[i]
                w_i = variable_value
                c_i = c_fixed
            elseif variable_name == :capital_price
                P_i = P_fixed[i]
                w_i = w_fixed
                c_i = variable_value
            else
                error("Invalid variable name")
            end
            # Calculate profit at time i
            profit_i = dt * discount_factors[i] * (
                P_i * Y_opt[i]
                - w_i * L_opt[i]
                - c_i * I_opt[i]
                - IP_opt[i]
                - CTP_opt[i]
            )
            total_profit += profit_i
        end
        return total_profit
    end

    # Function to find zero-profit condition for a variable
    function find_zero_profit(variable_name::Symbol, initial_guess::Float64)
        # Define the function whose root we want to find
        f!(F, x) = F[1] = calculate_profit(variable_name, x[1])

        # Use NLsolve to find the root
        result = nlsolve(f!, [initial_guess])
        zero_profit_value = result.zero[1]
        return zero_profit_value
    end


    # Perform partial equilibrium analysis for each variable
    zero_profit_price = find_zero_profit(:price, P_fixed[1])  # Initial guess is current price
    zero_profit_wage = find_zero_profit(:wage, w_fixed)       # Initial guess is current wage
    zero_profit_capital_price = find_zero_profit(:capital_price, c_fixed)  # Initial guess is current capital price

    # Collect the results
    zero_profit_conditions = Dict(
        "zero_profit_price" => zero_profit_price,
        "zero_profit_wage" => zero_profit_wage,
        "zero_profit_capital_price" => zero_profit_capital_price
    )

    return zero_profit_conditions
end
