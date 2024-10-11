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

def get_desired_price(expected_price, desired_price, real_price, actual_sales, desired_sales, min_price, optimal_inventory, inventory, production_gap):

    real_price = max(real_price, min_price)
    desired_price = max(desired_price, min_price)

    # Ratios and deviations
    if production_gap<0:
      production_gap = 0
    sales_ratio = actual_sales / (desired_sales - production_gap) if desired_sales - production_gap > 0 else  1.0
    inventory_ratio = inventory / (optimal_inventory - production_gap) if optimal_inventory - production_gap > 0 else 1.0

    # Deviations from the desired values
    sales_deviation = sales_ratio - 1.0         # Negative if sales < desired
    inventory_deviation = inventory_ratio - 1.0 # Positive if inventory > optimal

    if abs(sales_deviation) < 0.2 and abs(inventory_deviation) < 0.2:
      adjustment_factor = np.random.uniform(0.99, 1.15)
    else :
         # Overall adjustment factor: negative when sales are low or inventory is high
         adjustment_factor = (sales_deviation - inventory_deviation)

    # Cap the adjustment factor to be between -1 and 1
    adjustment_factor = max(-1.0, min(adjustment_factor, 2)) + np.random.uniform(-0.05, 0.15)

    # Scaling factor controls the magnitude of adjustment
    if adjustment_factor < 0:
      scaling_factor = 0.05
    else:
      scaling_factor = 0.25


    # Adjust the desired price directly based on the adjustment factor
    desired_price += desired_price * adjustment_factor * scaling_factor

    # Smooth the price to avoid abrupt changes
    smoothed_price = desired_price * 0.6 + real_price * 0.4

    # Ensure the smoothed price is not below the minimum price
    smoothed_price = max(smoothed_price, min_price)

    return smoothed_price

def get_desired_wage(expected_wage, desired_wage, real_wage, optimal_labor, actual_labor,max_wage, min_wage):
  real_wage = max(min_wage, min(real_wage, max_wage))
  desired_wage = max(min_wage,min(desired_wage, max_wage))
  labor_ratio = actual_labor/ optimal_labor if optimal_labor > 0 else 0
  adjustment_factor = 1
  if labor_ratio < 0.9:
    adjustment_factor = 0.98
  elif labor_ratio > 1.1:
    adjustment_factor = 1.02
  adjustment_factor += np.random.uniform(-0.05, 0.05)
  desired_wage = desired_wage * adjustment_factor

  desired_wage =min(max_wage, max(desired_wage, min_wage))

  smoothed_wage = desired_wage*0.6 + real_wage*0.4

  return smoothed_wage


## Not adjusting fast enough
def update_worker_price_expectation(price_decision_data):
    # Extract data
    expected_price = price_decision_data['expected_price']
    real_price = price_decision_data['real_price']
    desired_price = price_decision_data['desired_price']
    desired_consumption = price_decision_data['desired_consumption']
    consumption = price_decision_data['consumption']
    max_price = price_decision_data['max_price']


    real_price = max(real_price, expected_price)
    max_price = max(max_price, desired_price, expected_price)
    desired_price = min(desired_price, max_price)


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
        # Even if price_gap is small, we need to adjust desired_price upwards
        adjustment_factor += (1.0 - consumption_ratio)*3
        # Optionally, add more weight if price_gap is small
        if abs(price_gap_ratio) < 0.05:
            # Price gap is less than 5%, increase adjustment
            adjustment_factor += (0.05 - abs(price_gap_ratio)) * 20  # Weight can be adjusted

    else:
        # Consuming desired amount or more
        if desired_price > real_price:
            # Can lower desired_price to save money
            price_gap = desired_price - real_price
            price_gap_ratio = price_gap / desired_price if desired_price > 0 else 0.0
            adjustment_factor -= price_gap_ratio
        else:
            # Desired_price is less than or equal to real_price; maintain or slightly increase
            adjustment_factor += 0.0  # No change needed

    # Cap adjustment_factor between -1 and 1
    adjustment_factor = max(-1.0, min(adjustment_factor, 2.0))

    # Scaling factor controls the magnitude of adjustment
    scaling_factor = 0.5  # Increased from 0.05 to allow for larger adjustments

    # Adjust the desired price based on the adjustment factor
    desired_price += desired_price * adjustment_factor * scaling_factor

    # Smooth the price to avoid abrupt changes
    smoothed_price = desired_price * 0.6 + real_price * 0.4 # Slightly favor desired_price more

    # Ensure the smoothed price does not exceed the maximum price
    smoothed_price = min(smoothed_price, max_price)

    return smoothed_price



def update_worker_wage_expectation(wage_decision_data):

  real_wage = wage_decision_data['real_wage']
  desired_wage = wage_decision_data['desired_wage']
  expected_wage = wage_decision_data['expected_wage']
  working_hours = wage_decision_data['working_hours']
  desired_working_hours = wage_decision_data['optimal_working_hours']
  min_wage = wage_decision_data['min_wage']

  real_wage = max(real_wage, min_wage)
  desired_wage = max(desired_wage, min_wage)

  working_hours_ratio = working_hours / desired_working_hours if desired_working_hours > 0 else 0

  if working_hours_ratio < 0.9:
    adjustment_factor = 0.98
  elif working_hours_ratio:
    adjustment_factor = 1.02
  else:
    if desired_wage > real_wage:
      adjustment_factor = 0.98
    else:
      adjustment_factor = 1.02
  adjustment_factor += np.random.uniform(-0.05, 0.05)
  desired_wage = desired_wage * adjustment_factor
  desired_wage = max(desired_wage, min_wage)
  smoothed_wage = desired_wage*0.7 + expected_wage*0.1 + real_wage*0.2
  return smoothed_wage


# function signature mismatch 
def calculate_new_price(current_price, clearing_prices, market_demand, market_supply, inventory, optimal_inventory, expected_future_price):
    # Ensure market_demand and market_supply are lists


    market_demand = [market_demand] if isinstance(market_demand, (int, float)) else market_demand
    market_supply = [market_supply] if isinstance(market_supply, (int, float)) else market_supply
    clearing_prices = [clearing_prices] if isinstance(clearing_prices, (int, float)) else clearing_prices
    expected_future_price = [expected_future_price] if isinstance(expected_future_price, (int, float)) else expected_future_price
    
    # Calculate market tightness
    market_tightness = market_demand[0] / market_supply[0] if market_supply[0] > 0 else 1

    # Estimate market clearing price and quantity
    eq_quantity, eq_price, max_willingness_to_pay = estimate_intersection(market_supply, market_demand, clearing_prices)

    # Set target price based on market tightness
    if eq_quantity is None or eq_price is None or max_willingness_to_pay is None:

        target_price = current_price

    else:

        if market_tightness > 1:
            target_price = current_price + (max_willingness_to_pay - current_price) * 0.2
        else:
            target_price = current_price + (eq_price - current_price) * 0.2

    # Adjust based on inventory
    inventory_ratio = inventory / optimal_inventory if optimal_inventory > 0 else 1
    if inventory_ratio > 1:
        target_price *= 0.95  # Lower price if we have excess inventory
    elif inventory_ratio < 0.5:
        target_price *= 1.05  # Raise price if we have low inventory

    # Incorporate future expectations
    target_price = 0.8 * target_price + 0.2 * expected_future_price[0]

    # Implement gradual adjustment (max 5% change per period)
    max_change = 0.05 * current_price

    new_price = max(min(target_price, current_price + max_change), current_price - max_change)

    # Final check to ensure new_price is not NaN
    if np.isnan(new_price):
        print("Warning: new_price is NaN, falling back to current_price")
        new_price = current_price

    return new_price

def estimate_intersection(supply, demand, clearing_prices):
    # Ensure all arrays are the same length
    if len(supply) != len(demand) or len(supply) != len(clearing_prices):
        raise ValueError("Supply, demand, and clearing_prices must be of equal length")

    try:
        # Fit linear regression for supply
        slope_supply, intercept_supply, r_value_supply, p_value_supply, std_err_supply = stats.linregress(supply, clearing_prices)
        
        max_supply = max(supply)
        max_willingness_to_pay = slope_supply * max_supply + intercept_supply

        # Fit linear regression for demand
        slope_demand, intercept_demand, r_value_demand, p_value_demand, std_err_demand = stats.linregress(demand, clearing_prices)
        
        # Calculate intersection point
        quantity_intersection = (intercept_demand - intercept_supply) / (slope_supply - slope_demand)
        price_intersection = slope_supply * quantity_intersection + intercept_supply

        # Check for valid intersection
        if quantity_intersection <= 0 or price_intersection <= 0:
            return None, None, max_willingness_to_pay

        return quantity_intersection, price_intersection, max_willingness_to_pay
    except Exception as e:
        print(f"Error in estimate_intersection: {str(e)}")
        return None, None, None