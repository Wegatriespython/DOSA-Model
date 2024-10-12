import numpy as np
from scipy import stats


def get_max_wage(total_working_hours, productivity, capital, capital_elasticity, price, total_labor_units, labor, minimum_wage):
    if total_working_hours < 16:
        production_capacity = calculate_production_capacity(productivity, capital, capital_elasticity, 1)
        revenue_per_hour = (production_capacity * price) / 16
        max_wage = revenue_per_hour
    else:
        labor_demand = labor
        new_units = labor_demand

        new_production_capacity = calculate_production_capacity(productivity, capital, capital_elasticity, new_units)

        extra_revenue = (new_production_capacity ) * price

        extra_revenue_per_hour = extra_revenue / 16
        max_wage = extra_revenue_per_hour

    return max(max_wage, minimum_wage)

def get_min_sale_price(firm_type, workers, productivity, capital, capital_elasticity, total_labor_units, inventory):
    if firm_type == 'consumption':
        labor_cost = sum([worker['wage'] * worker['hours'] for worker in workers.values()])
        capital_cost = 0
        total_cost = labor_cost + capital_cost
        total_output = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units) + inventory
        if total_output <= 0 or total_cost <= 0.001:
            return 0.5
        return max(total_cost / total_output, 0.5)
    else:
        total_working_hours = sum([worker['hours'] for worker in workers.values()])
        average_wage = sum([worker['wage'] * worker['hours'] for worker in workers.values()]) / total_working_hours if total_working_hours > 0 else 0
        labor_cost = total_working_hours * average_wage
        total_output = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units) + inventory
        if total_output <= 0 or labor_cost <= 0.001:
            return 0.5
        return max(labor_cost / total_output,0.5)

def get_max_capital_price(investment_demand, optimal_production, optimal_capital, price, capital_elasticity, time_horizon, discount_rate):


    optimal_price = price


    total_revenue = sum([optimal_production * optimal_price * (1 - discount_rate)**i for i in range(time_horizon)])

    marginal_revenue_product = (total_revenue / optimal_capital) * capital_elasticity

    max_price_factor = 1.2
    max_capital_price = marginal_revenue_product * max_price_factor

    return max_capital_price

def calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units):
    production_capacity = productivity *(capital ** capital_elasticity)* total_labor_units ** (1 - capital_elasticity)
    return production_capacity
def get_desired_capital_price(self):
    capital_price = self.price * 3
    return capital_price

def get_desired_price(price_params):
    expected_price = price_params['expected_price']
    desired_price = price_params['desired_price']
    real_price = price_params['real_price'] 
    actual_sales = price_params['actual_sales']
    desired_sales = price_params['desired_sales']
    inventory = price_params['inventory']
    optimal_inventory = price_params['optimal_inventory']
    min_price = price_params['min_price']
    production_gap = price_params['production_gap']
    a_round = price_params['a_round']
    market_advantage = price_params['market_advantage']
    max_price = price_params['max_price']
    og_desired_price = desired_price

    # Handle case when no transactions occurred
    if market_advantage == "failure":
        # Significant price cut when no transactions occur
        price_cut_factor = 0.8  # 20% price cut
        desired_price *= price_cut_factor
        return max(min(desired_price, max_price), min_price)

    # Rest of the function remains the same
    # Ratios and deviations
    sales_ratio = actual_sales / desired_sales if desired_sales > 0 else 1.0
    inventory_ratio = inventory / optimal_inventory if optimal_inventory > 0 else 1.0

    # Deviations from the desired values
    sales_deviation = sales_ratio - 1.0
    inventory_deviation = inventory_ratio - 1.0

    # Base adjustment factor
    adjustment_factor = (sales_deviation - inventory_deviation)

    # Exploratory factor: allows for price increases even when sales are good
    exploratory_factor = np.random.uniform(-0.1, 0.2)

    # Combine adjustment and exploratory factors
    combined_factor = adjustment_factor + exploratory_factor

    # Cap the combined factor
    combined_factor = max(-0.2, min(combined_factor, 0.3))

    # Scaling factor controls the magnitude of adjustment
    scaling_factor = 0.1 if a_round == 1 else 0.2

    # Adjust the desired price
    desired_price += desired_price * combined_factor * scaling_factor

    # If in round 2 with seller advantage, be more aggressive
    if a_round == 2 and market_advantage == 'seller':
        desired_price *= 1.1
    elif a_round == 2 and market_advantage == 'buyer':
        desired_price *= 0.9

    # Smooth the price to avoid abrupt changes
    smoothed_price = desired_price * 0.7 + real_price * 0.3

    # Ensure the smoothed price is not below the minimum price
    smoothed_price = max(min(smoothed_price, max_price), min_price)
    return smoothed_price   

def get_desired_wage(wage_params):
    expected_wage = wage_params['expected_wage']
    desired_wage = wage_params['desired_wage']
    real_wage = wage_params['real_wage']
    optimal_labor = wage_params['optimal_labor']
    actual_labor = wage_params['actual_labor']
    max_wage = wage_params['max_wage']
    min_wage = wage_params['min_wage']
    a_round = wage_params['a_round']
    market_advantage = wage_params['market_advantage']

    # Ensure wages are within bounds
    real_wage = max(min_wage, min(real_wage, max_wage))
    desired_wage = max(min_wage, min(desired_wage, max_wage))

    # Calculate labor ratio
    labor_ratio = actual_labor / optimal_labor if optimal_labor > 0 else 0

    # Base adjustment factor
    if labor_ratio < 0.9:
        adjustment_factor = 0.98  # Lower wage if not enough work
    elif labor_ratio > 1.1:
        adjustment_factor = 1.02  # Increase wage if overworked
    else:
        adjustment_factor = 1.00  # Maintain wage if work is about right

    # Exploratory factor
    exploratory_factor = np.random.uniform(-0.05, 0.05)

    # Combine adjustment and exploratory factors
    combined_factor = adjustment_factor + exploratory_factor

    # Adjust desired wage
    desired_wage *= combined_factor

    # Round-specific adjustments
    if a_round == 1:
        # In round 1, be slightly more conservative
        desired_wage = desired_wage * 0.9 + real_wage * 0.1
    else:  # Round 2
        if market_advantage == 'seller':
            # If sellers (workers) have advantage, be more aggressive
            desired_wage = max(desired_wage, real_wage) * 1.05
        else:
            # If buyers (firms) have advantage, be more conservative
            desired_wage = min(desired_wage, real_wage) * 0.98

    # Ensure desired wage is within bounds
    desired_wage = min(max_wage, max(desired_wage, min_wage))

    # Smooth the wage changes
    smoothed_wage = desired_wage * 0.6 + real_wage * 0.3 + expected_wage * 0.1

    return smoothed_wage



def update_worker_price_expectation(price_decision_data):
    # Extract data
    expected_price = price_decision_data['expected_price']
    real_price = price_decision_data['real_price']
    desired_price = price_decision_data['desired_price']
    desired_consumption = price_decision_data['desired_consumption']
    consumption = price_decision_data['consumption']
    max_price = price_decision_data['max_price']
    a_round = price_decision_data['a_round']
    market_advantage = price_decision_data['market_advantage']

    if not a_round:
        desired_price *= 1.2
        return max(desired_price, max_price)
    # Ensure prices are within bounds
    real_price = max(real_price, expected_price)
    desired_price = min(desired_price, max_price)
    max_price = max(max_price, desired_price, real_price)
    
    # Compute consumption ratio
    consumption_ratio = consumption / desired_consumption if desired_consumption > 0 else 1.0

    # Deviations
    consumption_deviation = consumption_ratio - 1.0  # Negative if consuming less than desired
    price_gap = real_price - desired_price
    price_gap_ratio = price_gap / real_price if real_price > 0 else 0.0  # Positive if underbidding

    # Initialize adjustment factor
    adjustment_factor = 0.0

    # If consuming less than desired
    if consumption_ratio < 1.0:
        # Need to increase desired_price to get more supply
        adjustment_factor += (1.0 - consumption_ratio) * 0.5
    else:
        # Consuming desired amount or more
        if desired_price > real_price:
            # Can lower desired_price to save money
            adjustment_factor -= price_gap_ratio * 0.5

    # Round-specific adjustments
    if a_round == 1:
        # Be more conservative in round 1
        adjustment_factor *= 0.5
    else:  # Round 2
        if market_advantage == 'buyer':
            # If buyers have advantage, be more aggressive in lowering price
            adjustment_factor -= 0.1
        else:
            # If sellers have advantage, be more willing to increase price
            adjustment_factor += 0.1

    # Exploratory factor
    exploratory_factor = np.random.uniform(-0.05, 0.05)

    # Combine adjustment and exploratory factors
    combined_factor = adjustment_factor + exploratory_factor

    # Cap combined_factor
    combined_factor = max(-0.2, min(combined_factor, 0.2))

    # Adjust the desired price
    desired_price *= (1 + combined_factor)


    # Smooth the price changes
    smoothed_price = desired_price * 0.6 + real_price * 0.3 + expected_price * 0.1

    # Ensure the smoothed price does not exceed the maximum price
    smoothed_price = min(smoothed_price, max_price)

    return smoothed_price


def update_worker_wage_expectation(wage_decision_data):
    expected_wage = wage_decision_data['expected_wage']
    desired_wage = wage_decision_data['desired_wage']
    real_wage = wage_decision_data['real_wage']
    working_hours = wage_decision_data['working_hours']
    desired_working_hours = wage_decision_data['desired_working_hours']
    min_wage = wage_decision_data['min_wage']
    a_round = wage_decision_data['a_round']
    market_advantage = wage_decision_data['market_advantage']


    # Ensure wages are within bounds
    real_wage = max(real_wage, min_wage)
    desired_wage = max(desired_wage, min_wage)

    if not a_round:
        desired_wage *= 1.05
        return max(desired_wage, min_wage)

    # Calculate working hours ratio
    working_hours_ratio = working_hours / desired_working_hours if desired_working_hours > 0 else 0

    # Base adjustment factor
    if working_hours_ratio < 0.9:
        adjustment_factor = -0.02  # Lower wage expectations if not enough work
    elif working_hours_ratio > 1.1:
        adjustment_factor = 0.02   # Increase wage expectations if overworked
    else:
        adjustment_factor = 0.0    # Maintain wage if work is about right

    # Round-specific adjustments
    if a_round == 1:
        # Be more conservative in round 1
        adjustment_factor *= 0.5
    else:  # Round 2
        if market_advantage == 'seller':
            # If sellers (workers) have advantage, be more aggressive
            adjustment_factor += 0.03
        else:
            # If buyers (firms) have advantage, be more conservative
            adjustment_factor -= 0.03

    # Exploratory factor
    exploratory_factor = np.random.uniform(-0.03, 0.03)

    # Combine adjustment and exploratory factors
    combined_factor = adjustment_factor + exploratory_factor

    # Cap combined_factor
    combined_factor = max(-0.1, min(combined_factor, 0.1))

    # Adjust desired wage
    desired_wage *= (1 + combined_factor)

    # Ensure desired wage is within bounds
    desired_wage = max(min_wage, desired_wage)

    # Smooth the wage changes
    smoothed_wage = desired_wage * 0.6 + real_wage * 0.3 + expected_wage * 0.1

    return smoothed_wage


