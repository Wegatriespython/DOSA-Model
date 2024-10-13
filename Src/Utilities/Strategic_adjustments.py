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




def best_response_exact(price_decision_data):
    """
    Compute the exact optimal bid or ask price for a player in the two-round market clearing mechanism,
    accounting for non-equilibrium market conditions.
    """
    is_buyer = price_decision_data['is_buyer']
    round_num = price_decision_data['round_num']
    avg_buyer_price = price_decision_data['avg_buyer_price']
    avg_seller_price = price_decision_data['avg_seller_price']
    std_buyer_price = price_decision_data['std_buyer_price']
    std_seller_price = price_decision_data['std_seller_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    expected_demand = price_decision_data['expected_demand']
    expected_supply = price_decision_data['expected_supply']
    previous_price = price_decision_data['previous_price']
    R_p = pvt_res_price

    # Market imbalance
    imbalance = expected_demand - expected_supply

    # Handle edge cases of zero supply or demand
    if expected_supply == 0:
        if is_buyer:
            return R_p  # Buyers willing to pay up to their reservation price
        else:
            return R_p * 2  # Sellers can ask any price; no competition
    if expected_demand == 0:
        if is_buyer:
            return 0  # Buyers will not bid
        else:
            return R_p  # Sellers accept their reservation price

    # Determine opponent's average price and standard deviation
    if is_buyer:
        mu_opponent = avg_seller_price
        sigma_opponent = std_seller_price
    else:
        mu_opponent = avg_buyer_price
        sigma_opponent = std_buyer_price

    # Adjustment cost factor may represent economic frictions
    adjustment_cost_factor = 0.1

    def adjustment_cost(price):
        return adjustment_cost_factor * (price - previous_price) ** 2

    def expected_utility(price):
        if is_buyer:
            P_transaction = norm.cdf((price - mu_opponent) / sigma_opponent)
            if P_transaction == 0:
                return 0
            expected_seller_ask = mu_opponent - sigma_opponent * norm.pdf((price - mu_opponent) / sigma_opponent) / P_transaction
            P_t = (price + expected_seller_ask) / 2
            U = (R_p - P_t) * P_transaction
        else:
            P_transaction = 1 - norm.cdf((price - mu_opponent) / sigma_opponent)
            if P_transaction == 0:
                return 0
            expected_buyer_bid = mu_opponent + sigma_opponent * norm.pdf((price - mu_opponent) / sigma_opponent) / P_transaction
            P_t = (price + expected_buyer_bid) / 2
            U = (P_t - R_p) * P_transaction

        # Subtract adjustment cost
        U -= adjustment_cost(price)

        # Adjust utility based on market imbalance
        if imbalance != 0:
            imbalance_factor = imbalance / (expected_demand + expected_supply)
            if is_buyer:
                # Buyers may need to bid higher in tight markets
                U += U * imbalance_factor
            else:
                # Sellers may raise prices when demand is high
                U += U * (-imbalance_factor)
        return -U  # Negative for minimization

    # Set optimization bounds
    if is_buyer:
        lower_bound = max(0, min(previous_price * 0.9, R_p))
        upper_bound = R_p
    else:
        lower_bound = R_p
        upper_bound = max(previous_price * 1.1, R_p)

    # Optimize the expected utility
    result = minimize_scalar(
        expected_utility,
        bounds=(lower_bound, upper_bound),
        method='bounded'
    )

    optimal_price = result.x
    return optimal_price


