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

def buyer_heuristic(price_decision_data, debug = False):
    """
    Compute a heuristic bid price for a buyer in the two-round market clearing mechanism, Think deeper. Rn in case of round 1, buyers are not pushing up prices to raise supply and curb demand. 
    """
    avg_seller_price = price_decision_data['avg_seller_price']
    avg_buyer_price = price_decision_data['avg_buyer_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    avg_price = price_decision_data['price']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    previous_price = price_decision_data['previous_price']
    buyer_max_price = price_decision_data['avg_buyer_max_price']
    seller_min_price = price_decision_data['avg_seller_min_price']

    # Calculate market imbalance factor
    total_volume = demand + supply
    imbalance_factor = (demand - supply) / total_volume if total_volume > 0 else 0
    #imbalance factor is +ve when demand > supply
    #imbalance factor is -ve when supply > demand
    # Calculate the initial price estimate
    leverage = buyer_max_price - pvt_res_price


    match avg_buyer_price, avg_seller_price, supply, demand, seller_min_price, pvt_res_price, buyer_max_price:
        case (avg_buyer_price, avg_seller_price, _, _, _, _, _) if avg_buyer_price >= avg_seller_price:
             min_price_estimate = avg_seller_price
             if debug:
                print("Round 1:avg_buyer_price >= avg_seller_price")
        case (_, _, supply, demand, _, _, _) if demand >= supply : 
            match buyer_max_price, avg_seller_price:
                case (x, y) if x >= y:
                    min_price_estimate = avg_seller_price 
                    if debug:
                        print("Round 2: Sellet Advantage, buyer_max_price >= avg_seller_price")
                case (x, y) if x < y:
                    min_price_estimate = buyer_max_price
                    if debug:
                        print("Round 2: Seller Advantage, buyer_max_price < avg_seller_price")
        case (_, _, supply, demand, _, _, _) if supply > demand :
            match seller_min_price, avg_buyer_price:
                case (x, y) if x < y:
                    min_price_estimate =avg_buyer_price
                    if debug:
                        print("Round 2: Buyer Advantage, seller_min_price < avg_buyer_price")
                case (x, y) if x >= y:
                    min_price_estimate =seller_min_price * 1.05 # 5% margin of safety to prevent crashing through the price floor. 
                    if debug:
                        print("Round 2: Seller Advantage, seller_min_price >= avg_buyer_price")
        case (_, _, _, _, _, _, _) :
            
            print("avg_buyer_price, avg_seller_price, supply, demand, seller_min_price, pvt_res_price, buyer_max_price:")
            print(avg_buyer_price, avg_seller_price, supply, demand, seller_min_price, pvt_res_price, buyer_max_price)
            print("no case matched!")
            breakpoint()
    """Adjust the price estimate based on market imbalance
    Okay now the price estimates accurately model the minimum price under market conditions. The optimal strategy for the buyer is to pay the lowest price that the seller will accept. 

    However under market imbalance there are cases: 
    Case 1: Demand > Supply, 
        Buyers need to bid up the price for supply to meet demand. 
    Case 2: Supply > Demand, 
        Buyers can bid down the price to curb over supply. 
    Case 3: Demand = Supply
        Buyers need to very carefully adjust prices down so that the market clears. Overshooting is very dangerous. 

    """
    match imbalance_factor:
        case x if x >= 0:
            # This case we are adjusting to an upper bound. 
            if debug:
                print(f"Market Imbalance: Demand > Supply, imbalance_factor: {imbalance_factor}, min_price_estimate: {min_price_estimate}")
            price_estimate = min_price_estimate + imbalance_factor * (pvt_res_price - min_price_estimate)
        case x if x < 0:
            # This case we are adjusting to a lower bound. avg_seller_min is the lower bound. 
            if debug:
                print(f"Market Imbalance: Supply > Demand, imbalance_factor: {imbalance_factor}, min_price_estimate: {min_price_estimate}")
            price_estimate = min_price_estimate + imbalance_factor * (seller_min_price - min_price_estimate)
        case _:
            print("imbalance_factor: ", imbalance_factor)
            print("no case matched!")
            breakpoint()

    # Blend with previous price to add stability
        
    target_price = max(price_estimate, seller_min_price)

    final_price = previous_price + 0.2 * (target_price - previous_price)
    if debug:
        print(f"target_price {target_price}, previous_price {previous_price}, final_price {final_price}")
        print(f"pvt_res_price: {pvt_res_price}, seller_min_price: {seller_min_price}, output: {final_price}")
    # Ensure the price is within allowed bounds
    return min(final_price, pvt_res_price)

def seller_heuristic(price_decision_data, debug = False):
    """
    Compute a heuristic ask price for a seller in the two-round market clearing mechanism.
    Safety factor adjustment when round 1 leads to price stalling as max price is avg_buyer_price not buyer_max_price. Firms should seek to push prices out of round 1 when imbalances are detected. 


    """
    avg_buyer_price = price_decision_data['avg_buyer_price']
    avg_seller_price = price_decision_data['avg_seller_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    avg_price = price_decision_data['price']
    previous_price = price_decision_data['previous_price']
    buyer_max_price = price_decision_data['avg_buyer_max_price']
    seller_min_price = price_decision_data['avg_seller_min_price']

    match avg_buyer_price, avg_seller_price, supply, demand, seller_min_price, pvt_res_price, buyer_max_price:
        case (avg_buyer_price, avg_seller_price, _, _, _, _, _) if avg_buyer_price > avg_seller_price:
             max_price_estimate = avg_buyer_price
             if debug:
                print("Round 1:avg_buyer_price > avg_seller_price")
        case (_, _, supply, demand, _, _, _) if demand >= supply : 
            match buyer_max_price, avg_seller_price:
                case (x, y) if x >= y:
                    max_price_estimate = buyer_max_price * 0.95 # 5% margin of safety to prevent flying above the price ceiling. 
                    if debug:
                        print("Round 2: Seller Advantage, buyer_max_price >= avg_seller_price")
                case (x, y) if x < y:
                    max_price_estimate = avg_seller_price
                    if debug:
                        print("Round 2: Seller Advantage, buyer_max_price < avg_seller_price")
        case (_, _, supply, demand, _, _, _) if supply > demand :
            match seller_min_price, avg_buyer_price:
                case (x, y) if x < y:
                    max_price_estimate = avg_buyer_price
                    if debug:
                        print("Round 2: Buyer Advantage, seller_min_price < avg_buyer_price")
                case (x, y) if x >= y:
                    max_price_estimate = seller_min_price
                    if debug:
                        print("Round 2: Seller Advantage, seller_min_price >= avg_buyer_price")
        case (_, _, _, _, _, _, _) :
            print("avg_buyer_price, avg_seller_price, supply, demand, seller_min_price, pvt_res_price, buyer_max_price:")
            print(avg_buyer_price, avg_seller_price, supply, demand, seller_min_price, pvt_res_price, buyer_max_price)
            print("no case matched!")
            breakpoint()

    # Calculate market imbalance factor
    total_volume = demand + supply
    imbalance_factor = (demand - supply) / total_volume if total_volume > 0 else 0
    safety_factor = max(0, min(1, 1 - imbalance_factor))
    #fixed
    match imbalance_factor:
        case x if x >= 0:
            # when excess demand imbalance_factor is +ve,  price is estimated down from max_price_estimate
            price_estimate = max_price_estimate + safety_factor * (buyer_max_price - max_price_estimate)
            # Stay within bounds for round 2 under seller advantage clearing prices are between [avg_buyer_price, buyer_max_price], max_price_estimate is bounded by buyer_max_price so that is our upper bound. 
            if debug:
                print("Market Imbalance: Demand > Supply")
                print(f"imbalance_factor: {imbalance_factor}, safety_factor: {safety_factor}, max_price_estimate: {max_price_estimate}, avg_buyer_price: {avg_buyer_price}, price_estimate: {price_estimate}")
        case x if x < 0:
            # when excess supply imbalance_factor is -ve,  price is estimated up from seller_min_price
            price_estimate = max_price_estimate + safety_factor * (seller_min_price - max_price_estimate)
            if debug:
                print("Market Imbalance: Supply > Demand")
                print(f"imbalance_factor: {imbalance_factor}, safety_factor: {safety_factor}, max_price_estimate: {max_price_estimate}, seller_min_price: {seller_min_price}, price_estimate: {price_estimate}")
        case _:
            print("imbalance_factor: ", imbalance_factor)
            print("no case matched!")
            breakpoint()



    target_price = min(price_estimate, buyer_max_price)

  
    final_price =  previous_price + 0.2 * (target_price - previous_price)


    if debug:
        print(f"output {final_price}, previous_price {previous_price}, target_price {target_price}, buyer_max_price {buyer_max_price}")

    # Ensure the price is within allowed bounds
    return max(final_price, pvt_res_price)

def best_response_exact(price_decision_data, debug = False):
    """
    Compute a heuristic bid or ask price for a player in the two-round market clearing mechanism,
    using a simplified approach based on market conditions and private reservation price.
    """
    is_buyer = price_decision_data['is_buyer']
    return buyer_heuristic(price_decision_data, debug) if is_buyer else seller_heuristic(price_decision_data, debug)





