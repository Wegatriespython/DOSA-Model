import numpy as np


def is_valid_number(x):
    return isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)

def check_valid_params(params):
    for key, value in params.items():
        if isinstance(value, (list, np.ndarray)):
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                print(f"Invalid value in {key}: {value}")
                return False
        elif np.isnan(value) or np.isinf(value):
            print(f"Invalid value for {key}: {value}")
            return False
    return True



def find_equilibrium_price(demand, supply, bid, ask, bid_max, ask_min):
    price = 0 
    match bid, ask, demand, supply:
        case bid, ask, _, _ if bid >= ask:
            price = (bid + ask) / 2
        case bid, ask, _, _ if bid < ask:
            if demand >= supply:
                if bid_max >= ask:
                    price = (bid_max + ask) / 2
                else:
                    price = bid_max
            else:
                if bid >= ask_min:
                    price = (bid + ask_min) / 2
                else:
                    price = ask_min
        case _:
            print("Weird stuff")
            price = (bid + ask) / 2

    return price

def best_response_exact(price_decision_data, debug = False):
    """
    Compute a heuristic bid or ask price for a player in the two-round market clearing mechanism,
    using a simplified approach based on market conditions and private reservation price.
    """
    scenario = determine_scenario(price_decision_data)
    print(f"scenario: {scenario}")
    
    is_buyer = price_decision_data['is_buyer']
    return best_response_scenario_buyer(price_decision_data, debug) if is_buyer else best_response_scenario_seller(price_decision_data, debug)


def determine_scenario(price_decision_data):
    """Determine the current scenario for a buyer or seller."""
    #market_type = price_decision_data['market_type']
    round_num = price_decision_data['round_num']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    buyer_price = price_decision_data['bid']
    seller_price = price_decision_data['ask']
    buyer_max_price = price_decision_data['bid_max']
    seller_min_price = price_decision_data['ask_min']
    
    #print("market_type", market_type)

    if demand > supply:
        market_balance = "ExcessDemand"
    elif supply > demand:
        market_balance = "ExcessSupply"
    else:
        market_balance = "Equilibrium"

    if round_num == 1:
        trade_condition = "Trade" if buyer_price >= seller_price else "NoTrade"
        return f"Round1_{market_balance}_{trade_condition}"
    else:  # Round 2
        if market_balance == "ExcessDemand":
            trade_condition = "Trade" if buyer_max_price >= seller_price else "NoTrade"
            return f"Round2_SellerAdv_{market_balance}_{trade_condition}"
        else:  # ExcessSupply or Equilibrium (Buyer Advantage)
            trade_condition = "Trade" if buyer_price >= seller_min_price else "NoTrade"
            return f"Round2_BuyerAdv_{market_balance}_{trade_condition}"


def best_response_scenario_buyer(price_decision_data, debug=False):
    scenario = determine_scenario(price_decision_data)
    
    avg_buyer_price = price_decision_data['bid']
    avg_seller_price = price_decision_data['ask']
    avg_buyer_max_price = price_decision_data['bid_max']
    avg_seller_min_price = price_decision_data['ask_min']
    pvt_res_price = price_decision_data['pvt_res_price']
    previous_price = price_decision_data['previous_price']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    
    def adjust_for_imbalance(base_price, upper_bound=None, lower_bound=None):
        imbalance_factor = (demand - supply) / (demand + supply)
        adjustment = 0.1 * abs(imbalance_factor) + 0.01  # Ensure some minimum adjustment
        if imbalance_factor >= 0:
            return min(base_price * (1 + adjustment), upper_bound or float('inf'))
        else:
            return max(base_price * (1 - adjustment), lower_bound or 0)
    
    def blend_with_previous(target_price):
        return previous_price + 0.2 * (target_price - previous_price)  # Increased adjustment speed
    
    match scenario:
        case "Round1_ExcessDemand_Trade" | "Round1_ExcessDemand_NoTrade":
            upper_bound = min(avg_buyer_max_price * 1.05, pvt_res_price)
            target_price = adjust_for_imbalance(max(avg_seller_price, avg_buyer_price), upper_bound=upper_bound)
        case "Round2_SellerAdv_ExcessDemand_Trade" | "Round2_SellerAdv_ExcessDemand_NoTrade":
            target_price = min(avg_buyer_max_price * 1.02, pvt_res_price)
        case "Round1_Equilibrium_Trade" | "Round1_Equilibrium_NoTrade" | "Round2_BuyerAdv_Equilibrium_Trade" | "Round2_BuyerAdv_Equilibrium_NoTrade":
            target_price = (avg_seller_price + avg_buyer_price) / 2  # Meet in the middle
        case "Round1_ExcessSupply_Trade" | "Round1_ExcessSupply_NoTrade":
            lower_bound = max(avg_seller_min_price, avg_buyer_price * 0.95)
            target_price = adjust_for_imbalance(avg_buyer_price, lower_bound=lower_bound)
        case "Round2_BuyerAdv_ExcessSupply_Trade" | "Round2_BuyerAdv_ExcessSupply_NoTrade":
            target_price = max(avg_seller_min_price * 1.02, avg_buyer_price * 0.98)
        case _:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    final_price = blend_with_previous(target_price)
 
    
    if debug:
        print(f"Scenario: {scenario}, Final Price: {final_price}")
    
    return final_price


def best_response_scenario_seller(price_decision_data, debug=False):
    scenario = determine_scenario(price_decision_data)
    
    avg_buyer_price = price_decision_data['bid']
    avg_seller_price = price_decision_data['ask']
    avg_buyer_max_price = price_decision_data['bid_max']
    avg_seller_min_price = price_decision_data['ask_min']
    pvt_res_price = price_decision_data['pvt_res_price']
    previous_price = price_decision_data['previous_price']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    
    def adjust_for_imbalance(base_price, lower_bound=None, upper_bound=None):
        imbalance_factor = (demand - supply) / (demand + supply)
        adjustment = 0.1 * abs(imbalance_factor) + 0.01  # Ensure some minimum adjustment
        if imbalance_factor >= 0:
            return min(base_price * (1 + adjustment), upper_bound or float('inf'))
        else:
            return max(base_price * (1 - adjustment), lower_bound or 0)
    
    def blend_with_previous(target_price):
        return previous_price + 0.2 * (target_price - previous_price)  # Increased adjustment speed
    
    match scenario:
        case "Round1_ExcessDemand_Trade" | "Round1_ExcessDemand_NoTrade":
            upper_bound = avg_buyer_max_price * 0.95
            target_price = adjust_for_imbalance(max(avg_seller_price, avg_buyer_price), upper_bound=upper_bound)
        case "Round2_SellerAdv_ExcessDemand_Trade" | "Round2_SellerAdv_ExcessDemand_NoTrade":
            target_price = avg_buyer_max_price * 0.98
        case "Round1_Equilibrium_Trade" | "Round1_Equilibrium_NoTrade" | "Round2_BuyerAdv_Equilibrium_Trade" | "Round2_BuyerAdv_Equilibrium_NoTrade":
            target_price = (avg_seller_price + avg_buyer_price) / 2  # Meet in the middle
        case "Round1_ExcessSupply_Trade" | "Round1_ExcessSupply_NoTrade":
            lower_bound = max(pvt_res_price, avg_seller_price * 0.95)
            target_price = adjust_for_imbalance(min(avg_seller_price, avg_buyer_price), lower_bound=lower_bound)
        case "Round2_BuyerAdv_ExcessSupply_Trade" | "Round2_BuyerAdv_ExcessSupply_NoTrade":
            target_price = max(avg_seller_min_price * 1.02, pvt_res_price)
        case _:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    final_price = blend_with_previous(target_price)
    
    if debug:
        print(f"Scenario: {scenario}, Final Price: {final_price}")
    
    return final_price

