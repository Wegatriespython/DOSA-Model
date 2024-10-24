"""
A cobb-douglas preference function to calculate preference adjustment demand based on seller's attributes
For example : Seller has a productivity and carbon_intensity score for their goods.
Buyer submits their substitition elasticities between productivity and carbon_intensity to have a consistent overall demand curve.
Market matching then excutes the preference function to return the preference adjusted demand from the raw demand. THen the market matching goes on as usual.
Since each sale introduces 4 parameters, two from the seller and two from the buyer, this will have to be exeucuted for every price viable transaction."""




def preference_function(quantity, quality1, quality2, preference):
    # Placeholder for the preference function
    match preference:
        case 1:
            quantity = quantity * (1 + quality1)  # Adjust demand based on seller's productivity
        case 2:
            quantity = quantity * (1 + quality2)  # Adjust demand based on seller's carbon_intensity
        case 3:
            quantity = quantity * (1 + quality1 + quality2)  # Adjust demand based on both seller's productivity and carbon_intensity
        case _:
            quantity = quantity  # No adjustment
    # This should be implemented to calculate the adjusted demand based on buyer's preferences and seller's attributes
    return quantity  # For now, just return the initial demand

def market_matching(buyers, sellers):
    def match_recursively(buyers, sellers, transactions, a_round, market_advantage):
        if not buyers or not sellers:
            return transactions

        # Sort buyers (descending) and sellers (ascending) by price
        buyers.sort(key=lambda x: x[1], reverse=True)
        sellers.sort(key=lambda x: x[1])

        if buyers[0][1] < sellers[0][1]:
            if a_round == 1:
                # Start round 2: swap prices
                total_demand = sum(b[0] for b in buyers)
                total_supply = sum(s[0] for s in sellers)

                if total_demand > total_supply:
                    # Sellers Advantage
                    new_buyers = [(b[0], b[3], b[2], b[1], b[4]) for b in buyers]  # b[3] is max price
                    new_sellers = [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in sellers]  # s[3] is min price
                    market_advantage = 'seller'  # Seller advantage
                else:
                    # Buyers Advantage
                    new_buyers = [(b[0], b[1], b[2], b[3], b[4]) for b in buyers]  # b[3] is max price
                    new_sellers = [(s[0], s[3], s[2], s[1], s[4], s[5]) for s in sellers]  # s[3] is min price
                    market_advantage = 'buyer'  # Buyer advantage
                return match_recursively(new_buyers, new_sellers, transactions, 2, market_advantage)
            else:
                return transactions  # No more matches possible

        clearing_price = (buyers[0][1] + sellers[0][1]) / 2
        try:
            adjusted_quantity = preference_function(buyers[0][0], sellers[0][4], sellers[0][5], buyers[0][4])
        except IndexError as e:
            print(f"IndexError occurred. Buyers: {buyers}, Sellers: {sellers}")
            raise e

        quantity = min(adjusted_quantity, sellers[0][0])

        # Store round and market_advantage in the transaction
        transactions.append((buyers[0][2], sellers[0][2], quantity, clearing_price, a_round, market_advantage))

        # Update quantities
        new_buyers = [(buyers[0][0] - quantity, buyers[0][1], buyers[0][2], buyers[0][3], buyers[0][4])] + buyers[1:] if buyers[0][0] > quantity else buyers[1:]

        new_sellers = [(sellers[0][0] - quantity, sellers[0][1], sellers[0][2], sellers[0][3], sellers[0][4], sellers[0][5])] + sellers[1:] if sellers[0][0] > quantity else sellers[1:]

        # Remove any buyers or sellers with zero quantity
        new_buyers = [b for b in new_buyers if b[0] > 0]
        new_sellers = [s for s in new_sellers if s[0] > 0]

        # Recursive call
        return match_recursively(new_buyers, new_sellers, transactions, a_round, market_advantage)

    return match_recursively(buyers, sellers, [], 1, 0)  # Start with round 1 and buyer advantage (0)
