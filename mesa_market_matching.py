def market_matching(buyers, sellers):
    def match_recursively(buyers, sellers, transactions):
        if not buyers or not sellers:
            return transactions

        # Sort buyers (descending) and sellers (ascending) by price
        buyers.sort(key=lambda x: x[1], reverse=True)
        sellers.sort(key=lambda x: x[1])

        if buyers[0][1] < sellers[0][1]:
            return transactions  # No more matches possible

        clearing_price = (buyers[0][1] + sellers[0][1]) / 2
        quantity = min(buyers[0][0], sellers[0][0])

        transactions.append((buyers[0][2], sellers[0][2], quantity, clearing_price))

        # Update quantities
        new_buyers = [(buyers[0][0] - quantity, buyers[0][1], buyers[0][2])] + buyers[1:] if buyers[0][0] > quantity else buyers[1:]
        new_sellers = [(sellers[0][0] - quantity, sellers[0][1], sellers[0][2])] + sellers[1:] if sellers[0][0] > quantity else sellers[1:]

        # Remove any buyers or sellers with zero quantity
        new_buyers = [b for b in new_buyers if b[0] > 0]
        new_sellers = [s for s in new_sellers if s[0] > 0]

        # Recursive call
        return match_recursively(new_buyers, new_sellers, transactions)

    return match_recursively(buyers, sellers, [])
