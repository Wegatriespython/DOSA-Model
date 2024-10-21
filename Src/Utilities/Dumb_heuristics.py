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

def determine_scenario(price_decision_data):
    """Determine the current scenario for a buyer or seller."""
    #market_type = price_decision_data['market_type']
    round_num = price_decision_data['round_num']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    buyer_price = price_decision_data['avg_buyer_price']
    seller_price = price_decision_data['avg_seller_price']
    buyer_max_price = price_decision_data['avg_buyer_max_price']
    seller_min_price = price_decision_data['avg_seller_min_price']
    
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
    """Determine the optimal policy for a buyer based on the current scenario."""
    scenario = determine_scenario(price_decision_data)
    
    avg_buyer_price = price_decision_data['avg_buyer_price']
    avg_seller_price = price_decision_data['avg_seller_price']
    avg_buyer_max_price = price_decision_data['avg_buyer_max_price']
    avg_seller_min_price = price_decision_data['avg_seller_min_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    previous_price = price_decision_data['previous_price']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    
    def adjust_for_imbalance(base_price, upper_bound=None, lower_bound=None):
        imbalance_factor = (demand - supply) / (demand + supply)
        # test value demand = 30, supply = 20, demand + supply = 50, imbalance_factor = 10/50 = 0.2
        if imbalance_factor >= 0:
            return base_price + imbalance_factor * (upper_bound - base_price)
        else:
            return base_price + imbalance_factor * (base_price - lower_bound)
    
    def blend_with_previous(target_price):
        return previous_price + 0.001 * (target_price - previous_price) # Since we are setting the target price to the limiting value, we need to make small adjustments. 
    
    match scenario:
        case "Round1_ExcessDemand_Trade" | "Round1_ExcessDemand_NoTrade" :
            # Seller will set price close to the ceiling. Buyers are competing for limited supply. Race to be the highest bidder. 
            # Probably need to test to ensure this is proportional to the imbalance
            upper_bound = min(avg_buyer_max_price*1.01, pvt_res_price)
            target_price = adjust_for_imbalance(max(avg_seller_price, avg_buyer_price), upper_bound= upper_bound)
            if debug:
                print(f"Scenario: {scenario}, Target Price: {target_price}, base_price: {max(avg_seller_price, avg_buyer_price)}, upper_bound: {upper_bound}, pvt_res_price: {pvt_res_price}, average_buyer_max_price: {avg_buyer_max_price}")
        # Not possible. Round 1 always trades.  Its round 2 that might not trade. 
        case "Round2_SellerAdv_ExcessDemand_Trade"| "Round2_SellerAdv_ExcessDemand_NoTrade" :
            target_price = min(avg_buyer_max_price * 1.01, pvt_res_price)
            if debug: 
                print(f"Scenario: {scenario}, Target Price: {target_price} ") 

        case "Round1_Equilibrium_Trade" | "Round1_Equilibrium_NoTrade" | "Round2_BuyerAdv_Equilibrium_Trade" | "Round2_BuyerAdv_Equilibrium_NoTrade":
            # This is very close to the equilibrium price, sellers will target avg_buyer_price and buyers will target avg_seller_price.
            target_price = avg_seller_price * 1.02
            if debug: 
                print(f"Scenario: {scenario}, Target Price: {target_price} ") 
        
        case "Round1_ExcessSupply_Trade" | "Round1_ExcessSupply_NoTrade":
            # In Round 2 with seller advantage and possible trade, bid close to the average seller price
            lower_bound = avg_seller_min_price # Less caution as the imbalance factor dampens the adjustment lowering risk of overshooting. 
            target_price = adjust_for_imbalance(avg_buyer_price, lower_bound= lower_bound)
            if debug: 
                print(f"Scenario: {scenario}, Target Price: {target_price}, base_price: {avg_buyer_price} lower_bound: {lower_bound} ") 
        # Round 2 Equilibrium is not possible? Or is it?
        case "Round2_BuyerAdv_ExcessSupply_Trade" | "Round2_BuyerAdv_ExcessSupply_NoTrade":
            target_price = avg_seller_min_price * 1.05 # More caution when directly approaching the floor. 
            if debug: 
                print(f"Scenario: {scenario}, Target Price: {target_price} ") 
        case _:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    final_price = blend_with_previous(target_price)

    final_price_true = min(final_price, pvt_res_price)
    # Ensure we don't exceed private reservation price
    if debug: 
        print(f"Scenario: {scenario}, Final Price: {final_price}, final_price_true: {final_price_true} ") 
    
    
    return final_price_true


def best_response_scenario_seller(price_decision_data, debug=False):
    """Determine the optimal policy for a seller based on the current scenario."""
    scenario = determine_scenario(price_decision_data)
    
    avg_buyer_price = price_decision_data['avg_buyer_price']
    avg_seller_price = price_decision_data['avg_seller_price']
    avg_buyer_max_price = price_decision_data['avg_buyer_max_price']
    avg_seller_min_price = price_decision_data['avg_seller_min_price']
    pvt_res_price = price_decision_data['pvt_res_price']
    previous_price = price_decision_data['previous_price']
    demand = price_decision_data['demand']
    supply = price_decision_data['supply']
    
    def adjust_for_imbalance(base_price, lower_bound = None, upper_bound = None):
        imbalance_factor = (demand - supply) / (demand + supply)
        if imbalance_factor >= 0:
            return base_price + imbalance_factor * (upper_bound - base_price)
            # imbalance_factor is +ve when demand > supply, so this will raise the price estimate. 
        else:
            return base_price + imbalance_factor * (base_price - lower_bound) # Base price is always higher than lower_bound.  so this will lower the price estimate. 
            # imbalance_factor is -ve when supply > demand, so this will lower the price estimate. 
    
    def blend_with_previous(target_price):
        return previous_price + 0.001 * (target_price - previous_price)
    
    match scenario:
        case "Round1_ExcessDemand_Trade" | "Round1_ExcessDemand_NoTrade":
            upper_bound = avg_buyer_max_price * 0.85
            target_price = adjust_for_imbalance(base_price=max(avg_seller_price, avg_buyer_price), upper_bound=upper_bound)

        case "Round2_SellerAdv_ExcessDemand_Trade" | "Round2_SellerAdv_ExcessDemand_NoTrade":
            target_price = avg_buyer_max_price * 0.95

        case "Round1_Equilibrium_Trade" | "Round1_Equilibrium_NoTrade" | "Round2_BuyerAdv_Equilibrium_Trade" | "Round2_BuyerAdv_Equilibrium_NoTrade":
            target_price = avg_buyer_price * 0.99

        case "Round1_ExcessSupply_Trade" | "Round1_ExcessSupply_NoTrade":
            lower_bound = pvt_res_price
            target_price = adjust_for_imbalance(base_price=min(avg_seller_price, avg_buyer_price), lower_bound=lower_bound)
        case "Round2_BuyerAdv_ExcessSupply_Trade" | "Round2_BuyerAdv_ExcessSupply_NoTrade":
            target_price = min(avg_seller_min_price*0.98, pvt_res_price)
        case _:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    final_price = blend_with_previous(target_price)
    final_price_true = max(final_price, pvt_res_price)  # Ensure we don't go below private reservation price
    
    if debug:
        print(f"Scenario: {scenario}, Final Price: {final_price}")
    
    return final_price_true

