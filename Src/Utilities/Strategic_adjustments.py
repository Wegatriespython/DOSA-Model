import numpy as np
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

def get_desired_price(expected_price, desired_price, real_price, actual_sales, desired_sales, min_price, optimal_inventory, inventory):

    real_price = max(real_price, min_price)
    desired_price = max(desired_price, min_price)

    # Ratios and deviations
    sales_ratio = actual_sales / desired_sales if desired_sales > 0 else 1.0
    inventory_ratio = inventory / optimal_inventory if optimal_inventory > 0 else 1.0

    # Deviations from the desired values
    sales_deviation = sales_ratio - 1.0         # Negative if sales < desired
    inventory_deviation = inventory_ratio - 1.0 # Positive if inventory > optimal

    # Overall adjustment factor: negative when sales are low or inventory is high
    adjustment_factor = sales_deviation - inventory_deviation

    # Cap the adjustment factor to be between -1 and 1
    adjustment_factor = max(-1.0, min(adjustment_factor, 1.0)) + np.random.uniform(-0.05, 0.15)

    # Scaling factor controls the magnitude of adjustment
    scaling_factor = 0.05

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


## Needs major fix. Something causes workers to adjust prices down randomly causing issues.
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
        adjustment_factor += (1.0 - consumption_ratio)
        # Optionally, add more weight if price_gap is small
        if abs(price_gap_ratio) < 0.05:
            # Price gap is less than 5%, increase adjustment
            adjustment_factor += (0.20 - abs(price_gap_ratio)) * 20  # Weight can be adjusted

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
    adjustment_factor = max(-1.0, min(adjustment_factor, 1.0))

    # Scaling factor controls the magnitude of adjustment
    scaling_factor = 0.5  # Increased from 0.05 to allow for larger adjustments

    # Adjust the desired price based on the adjustment factor
    desired_price += desired_price * adjustment_factor * scaling_factor

    # Smooth the price to avoid abrupt changes
    smoothed_price = desired_price * 0.9 + real_price * 0.1 # Slightly favor desired_price more

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
