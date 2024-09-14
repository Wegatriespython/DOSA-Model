def get_max_wage(total_working_hours, productivity, capital, capital_elasticity, price, total_labor_units, optimals, minimum_wage):
    if total_working_hours < 16:
        production_capacity = calculate_production_capacity(productivity, capital, capital_elasticity, 1)
        revenue_per_hour = (production_capacity * price) / 16
        max_wage = revenue_per_hour
    else:
        labor_demand = optimals[0]
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
            return 1.5
        return max(labor_cost / total_output,1.5)

def get_max_capital_price(investment_demand, optimals, price, capital_elasticity, time_horizon, discount_rate):

    optimal_production = optimals[2]
    optimal_price = price
    optimal_capital = optimals[1]

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

def get_desired_price (expected_price, desired_price, real_price, actual_sales,desired_sales,min_price, optimal_inventory, inventory):

  real_price = max(real_price, min_price)
  desired_price = max(desired_price, min_price)

  sales_ratio = actual_sales / desired_sales if desired_sales > 0 else 0

  inventory_ratio = inventory / optimal_inventory if optimal_inventory > 0 else float('inf')

  adjustment_factor = 1.0
  if sales_ratio < 0.9:
      adjustment_factor = 0.98
  elif sales_ratio > 1.1:
      adjustment_factor = 1.02
  else:
      if inventory_ratio > 1.2:
          adjustment_factor = 0.99
      elif inventory_ratio < 0.8:
          adjustment_factor = 1.01
  new_price = desired_price * adjustment_factor
  new_price = max(new_price, min_price)
  smoothed_price = desired_price * 0.7 + real_price * 0.3  #
  return smoothed_price + (new_price - smoothed_price) * 0.2

def get_desired_wage(expected_wage, desired_wage, real_wage, optimal_labor, actual_labor,max_wage, min_wage):
  real_wage = max(min_wage, min(real_wage, max_wage))
  desired_wage = max(min_wage,min(desired_wage, max_wage))
  labor_ratio = actual_labor/ optimal_labor if optimal_labor > 0 else 0
  adjustment_factor = 1
  if labor_ratio < 0.9:
    adjustment_factor = 0.98
  elif labor_ratio > 1.1:
    adjustment_factor = 1.02
  desired_wage = desired_wage * adjustment_factor

  desired_wage =min(max_wage, max(desired_wage, min_wage))

  smoothed_wage = desired_wage*0.6 + real_wage*0.4

  return smoothed_wage

def update_worker_price_expectation(latent_price,desired_price,real_price, desired_consumption, consumption, max_price):
        real_price = max(real_price,latent_price)
        if consumption < desired_consumption:
          # If the worker is consuming less than desired, increase the expected price
          desired_price = min((real_price + (real_price -desired_price) * 0.5), max_price)
          return desired_price

        elif desired_price < real_price:
          # if consuming sufficient yet, overpaying, then round 2 clearing is happening, worker needs to increase bid to lower prices.
           desired_price = real_price - (real_price - desired_price) * 0.5
           smoothened_price = desired_price * 0.6 + real_price * 0.4
           return smoothened_price
        else:
          # if consuming and in round1 then worker can bargain by lowering the bid
           desired_price = real_price - (desired_price - real_price) * 0.2
           smoothened_price = desired_price * 0.6 + real_price * 0.4
           return smoothened_price




def update_worker_wage_expectation(expected_wage,desired_wage,real_wage, working_hours, desired_working_hours, min_wage):

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

  desired_wage = desired_wage * adjustment_factor
  desired_wage = max(desired_wage, min_wage)
  smoothed_wage = desired_wage*0.7 + expected_wage*0.1 + real_wage*0.2
  return smoothed_wage
