# Make sure to include your original file
include("V:\\Python Port\\Src\\Utilities\\Continous_Profit_Maxxin.jl")

# Test function
function test_profit_maximization()
    # Input parameters
    current_capital = 100.0
    current_labor = 50.0
    current_price = 10.0
    current_productivity = 1.0
    expected_demand = fill(100.0, 50)  # Vector of 50 elements, all 100
    expected_price = fill(10.0, 50)  # Vector of 50 elements, all 10.0
    capital_price = 5.0
    capital_elasticity = 0.3
    current_inventory = 20.0
    depreciation_rate = 0.1
    expected_periods = 10.0
    discount_rate = 0.05
    budget = 1000.0
    wage = 8.0
    capital_supply = 200.0
    labor_supply = 100.0

    # Call the function
    result = profit_maximization(
        current_capital, current_labor, current_price, current_productivity,
        expected_demand, expected_price, capital_price, capital_elasticity,
        current_inventory, depreciation_rate, expected_periods, discount_rate,
        budget, wage, capital_supply, labor_supply, 0.0, 0.0, 0.0, 0.0, 0.0
    )

    # Print the result
    println("Test completed. Result:")
    capital= result["optimal_capital"][1]
    labor= result["optimal_labor"][1]
    inventory= result["optimal_inventory"][1]
    Select_results= Dict("capital"=>capital, "labor"=>labor, "inventory"=>inventory)

    println("Select_results: ", Select_results)



end

# Run the test
test_profit_maximization()
