import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize_scalar



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

    return max(min(max_wage, 1.2), minimum_wage)

def get_min_sale_price(firm_type, workers, productivity, capital, capital_elasticity, total_labor_units, inventory):
    if firm_type == 'consumption':
        labor_cost = sum([worker['wage'] * worker['hours'] for worker in workers.values()])
        capital_cost = 0
        total_cost = labor_cost + capital_cost
        total_output = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units) + inventory
        if total_output <= 0 or total_cost <= 0.001:
            return 0.7
        return max(total_cost / total_output, 0.7)
    else:
        total_working_hours = sum([worker['hours'] for worker in workers.values()])
        average_wage = sum([worker['wage'] * worker['hours'] for worker in workers.values()]) / total_working_hours if total_working_hours > 0 else 0
        labor_cost = total_working_hours * average_wage
        total_output = calculate_production_capacity(productivity, capital, capital_elasticity, total_labor_units) + inventory
        if total_output <= 0 or labor_cost <= 0.001:
            return 0.7
        return max(labor_cost / total_output,0.7)

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


def buyer_heuristic(price_decision_data):
    """
    Compute a heuristic bid price for a buyer in the two-round market clearing mechanism.
    """
    avg_seller_price = price_decision_data['avg_seller_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    avg_price = price_decision_data['avg_price']
    expected_demand = price_decision_data['expected_demand']
    expected_supply = price_decision_data['expected_supply']
    previous_price = price_decision_data['previous_price']
    buyer_max_price = price_decision_data.get('buyer_max_price', pvt_res_price)
    seller_min_price = price_decision_data['seller_min_price']

    # Calculate market imbalance factor
    total_volume = expected_demand + expected_supply
    imbalance_factor = (expected_demand - expected_supply) / total_volume if total_volume > 0 else 0

    # Calculate the initial price estimate
    price_estimate = avg_seller_price - 0.1 * (avg_seller_price - seller_min_price)

    # Adjust the price estimate based on market imbalance
    price_estimate -= imbalance_factor * (price_estimate - pvt_res_price)

    # Blend with previous price to add stability
    price_estimate = 0.2 * price_estimate + 0.7 * previous_price + 0.1 * avg_price

    # Ensure the price is within allowed bounds
    return max(seller_min_price, min(price_estimate, pvt_res_price))

def seller_heuristic(price_decision_data):
    """
    Compute a heuristic ask price for a seller in the two-round market clearing mechanism.
    """
    avg_buyer_price = price_decision_data['avg_buyer_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    expected_demand = price_decision_data['expected_demand']
    expected_supply = price_decision_data['expected_supply']
    avg_price = price_decision_data['avg_price']
    previous_price = price_decision_data['previous_price']
    buyer_max_price = price_decision_data['buyer_max_price']
    seller_min_price = price_decision_data.get('seller_min_price', pvt_res_price)

    # Calculate market imbalance factor
    total_volume = expected_demand + expected_supply
    imbalance_factor = (expected_demand - expected_supply) / total_volume if total_volume > 0 else 0

    # Calculate the initial price estimate
    price_estimate = avg_buyer_price + 0.1 * (buyer_max_price - avg_buyer_price)

    # Adjust the price estimate based on market imbalance
    price_estimate += imbalance_factor * (buyer_max_price - price_estimate)

    # Blend with previous price to add stability
    price_estimate = 0.2 * price_estimate + 0.7 * previous_price + 0.1 * avg_price

    # Ensure the price is within allowed bounds
    return max(pvt_res_price, min(price_estimate, buyer_max_price))

def best_response_exact(price_decision_data):
    """
    Compute a heuristic bid or ask price for a player in the two-round market clearing mechanism,
    using a simplified approach based on market conditions and private reservation price.
    """
    is_buyer = price_decision_data['is_buyer']
    return buyer_heuristic(price_decision_data) if is_buyer else seller_heuristic(price_decision_data)





