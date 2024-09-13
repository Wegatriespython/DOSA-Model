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
        print('max_wage', max_wage)
    return max(max_wage, minimum_wage)

def get_min_sale_price(firm_type, workers, productivity, capital, capital_elasticity, total_labor_units, inventory):
    if firm_type == 'consumption':
        labor_cost = sum([worker['wage'] * worker['hours'] for worker in workers.values()])
        capital_cost = 0
        total_cost = labor_cost + capital_cost
        total_output = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units) + inventory
        if total_output <= 0 or total_cost <= 0.001:
            return 0.5
        return total_cost / total_output
    else:
        total_working_hours = sum([worker['hours'] for worker in workers.values()])
        average_wage = sum([worker['wage'] * worker['hours'] for worker in workers.values()]) / total_working_hours if total_working_hours > 0 else 0
        labor_cost = total_working_hours * average_wage
        total_output = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units) + inventory
        if total_output <= 0 or labor_cost <= 0.001:
            return 1.5
        return labor_cost / total_output

def get_max_capital_price(investment_demand, optimals, price, capital_elasticity, time_horizon, discount_rate):
    if investment_demand <= 0:
        return 0

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
    average_capital_price = self.model.data_collector.get_average_capital_price(self.model)
    return average_capital_price

def get_desired_wage(desired_wage,desired_labor, actual_labor, max_wage, real_wage, min_wage):

  if  actual_labor < desired_labor :
      wage = real_wage + (max_wage - real_wage)* 0.2
      print(f"increasing wage to {wage}")
  else :
      wage = real_wage - (real_wage - min_wage) * 0.2
  return wage

def get_desired_price(desired_price, desired_sales, actual_sales, min_price, real_price):
      if desired_price < real_price:
        price = desired_price + (real_price - desired_price) * 0.2
        return price

      else:
        print(f"Cutting prices{desired_price} real{real_price}")
        price = desired_price - (desired_price - min_price) * 0.2
        return price
