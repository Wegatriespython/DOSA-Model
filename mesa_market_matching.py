def market_matching(buyers, sellers):
    """
    Match buyers and sellers based on their demand/supply and max/min prices.
    
    :param buyers: List of tuples (demand, max_price, buyer_object)
    :param sellers: List of tuples (supply, min_price, seller_object)
    :return: List of successful transactions (buyer, seller, quantity, price)
    """
    # Sort buyers by descending max price and sellers by ascending min price
    buyers.sort(key=lambda x: x[1], reverse=True)
    sellers.sort(key=lambda x: x[1])
    
    transactions = []
    
    for buyer in buyers:
        buyer_demand, max_price, buyer_obj = buyer
        if buyer_demand <= 0:
            continue
        
        for seller in sellers:
            seller_supply, min_price, seller_obj = seller
            if seller_supply <= 0:
                continue
            
            if min_price <= max_price:
                quantity = min(buyer_demand, seller_supply)
                price = (max_price + min_price) / 2  # Set price as average of max and min
                
                transactions.append((buyer_obj, seller_obj, quantity, price))
                
                buyer_demand -= quantity
                seller_supply -= quantity
                
                # Update the original tuples
                buyer = (buyer_demand, max_price, buyer_obj)
                seller = (seller_supply, min_price, seller_obj)
                
                if buyer_demand == 0:
                    break
            else:
                break  # No more sellers with acceptable prices for this buyer
    
    return transactions