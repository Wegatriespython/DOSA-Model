def get_max_wage(total_working_hours, productivity, capital, capital_elasticity, price, total_labor_units, optimals, minimum_wage):
    if total_working_hours < 16:
        production_capacity = calculate_production_capacity(productivity, capital, capital_elasticity, 1)
        revenue_per_hour = (production_capacity * price) / 16
        max_wage = revenue_per_hour
    else:
        labor_demand = optimals[0]
        new_units = labor_demand / 16
        production_capacity = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units)
        new_production_capacity = calculate_production_capacity(productivity, capital, capital_elasticity, new_units)
        extra_revenue = (new_production_capacity - production_capacity) * price
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
def get_desired_wage(self, labor_supply, labor_demand):
  labor_wanted = self.optimals[0] * 16
  wage = self.wage
  max_wage = self.zero_profit_conditions[0]
  min_wage = self.model.config.MINIMUM_WAGE
  payout_optimal = labor_wanted * wage
  payout_max = labor_wanted * max_wage
  # Pay less when there is excess labor in the market. Pay more when there is a labor shortage.
  historical_labor_demand = self.optimals_cache[0]
  # If labor supply is greater than labor demand, reduce wage
  #
  print (f"Supply: {labor_supply}, Demand: {labor_demand}")

  if labor_supply > labor_demand:
      wage = min_wage + (wage - min_wage) * 0.2

  if self.get_total_labor_units() < historical_labor_demand[0]:
      wage = wage + (max_wage - wage)* 0.2
  else:
      wage = max(min_wage, wage * 0.95)

  print(f"Desired wage: {wage}")
  return wage
def get_desired_price(self, supply, demand, price, min_price):
    if supply > demand:
        price = price - (price - min_price) * 0.2
        return price
    else:
        price = price * 1.05

        return price
