import os,sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, src_path)
from Utilities.mesa_market_matching import market_matching, preference_function

def test_market_matching():
    # Case 1: Buyers have a higher max price than sellers' min price
    buyers_case1 = [(100, 70, f"Buyer_{i}", 75) for i in range(30)]  # (quantity, bid, id, max_price)
    sellers_case1 = [(1000, 80, f"Seller_{i}", 70) for i in range(5)]  # (quantity, ask, id, min_price)

    # Case 2: Buyers have a lower max price than sellers' min price
    buyers_case2 = [(100, 70, f"Buyer_{i}", 80) for i in range(30)]  # (quantity, bid, id, max_price)
    sellers_case2 = [(1000, 90, f"Seller_{i}", 85) for i in range(5)]  # (quantity, ask, id, min_price)

    print("Case 1: Buyers have a higher max price than sellers' min price")
    transactions_case1 = market_matching(buyers_case1, sellers_case1)
    print(f"Number of transactions: {len(transactions_case1)}")
    for t in transactions_case1[:5]:  # Print first 5 transactions
        print(f"Buyer: {t[0]}, Seller: {t[1]}, Quantity: {t[2]}, Price: {t[3]}")
    print("...")

    print("\nCase 2: Buyers have a lower max price than sellers' min price")
    transactions_case2 = market_matching(buyers_case2, sellers_case2)
    print(f"Number of transactions: {len(transactions_case2)}")
    for t in transactions_case2[:5]:  # Print first 5 transactions
        print(f"Buyer: {t[0]}, Seller: {t[1]}, Quantity: {t[2]}, Price: {t[3]}")
    print("...")

if __name__ == "__main__":
    test_market_matching()
