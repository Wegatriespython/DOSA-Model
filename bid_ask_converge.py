import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of 'Src' to the Python path
src_parent_dir = os.path.dirname(current_dir)
sys.path.append(src_parent_dir)

from Src.Utilities.market_matching import market_matching, preference_function
from Src.Utilities.Strategic_adjustments import best_response_scenario_buyer, best_response_scenario_seller
from Src.Utilities.Profit_maximization import profit_maximization
from Src.Utilities.utility_function import maximize_utility
from Src.Utilities.adaptive_expectations import adaptive_expectations

# Functions for demand and supply curves
def demand_curve(price, a, b):
    """Linear demand curve: Q = a - b*P"""
    return max(0, a - b * price)

def supply_curve(price, c, d):
    """Linear supply curve: Q = c + d*P"""
    return max(0, c + d * price)

# Parameters
num_buyers = 10
num_sellers = 10
max_iterations = 50
tolerance = 1e-3  # Convergence tolerance

# Demand and supply curve parameters
a, b = 100, 1  # Demand curve: Q = 100 - P
c, d = 0, 1    # Supply curve: Q = P

# Initialize buyers and sellers with reservation prices
np.random.seed(0)  # For reproducibility

# Buyers' maximum willingness to pay (reservation prices)
B_max = np.random.uniform(50, 100, num_buyers)
# Sellers' minimum willingness to accept (reservation prices)
A_min = np.random.uniform(0, 50, num_sellers)

# Initial bids and asks
B = np.random.uniform(40, 60, num_buyers)
A = np.random.uniform(40, 60, num_sellers)

# Lists to store average bid and ask prices over time
avg_bid_history = []
avg_ask_history = []
transaction_prices = []

for t in range(max_iterations):
    print(f"\nIteration {t+1}")
    
    # Record average bid and ask prices
    avg_bid = np.mean(B)
    avg_ask = np.mean(A)
    avg_bid_history.append(avg_bid)
    avg_ask_history.append(avg_ask)
    
    print(f"Average Bid: {avg_bid:.2f}, Average Ask: {avg_ask:.2f}")

    # Calculate demand and supply based on current price
    current_price = (avg_bid + avg_ask) / 2
    current_demand = demand_curve(current_price, a, b)
    current_supply = supply_curve(current_price, c, d)
    
    print(f"Current Price: {current_price:.2f}, Demand: {current_demand:.2f}, Supply: {current_supply:.2f}")

    # Prepare price decision data for strategic adjustments
    price_decision_data = {
        'avg_buyer_price': avg_bid,
        'avg_seller_price': avg_ask,
        'avg_buyer_max_price': np.mean(B_max),
        'avg_seller_min_price': np.mean(A_min),
        'demand': current_demand,
        'supply': current_supply,
        'price': current_price,
        'previous_price': avg_bid_history[-2] if len(avg_bid_history) > 1 else avg_bid,
        'market_type': 'consumption',
        'round_num': 1
    }

    # Update bids and asks using best response functions
    new_B = []
    new_B_max = []
    for i in range(num_buyers):
        price_decision_data['pvt_res_price'] = B_max[i]
        price_decision_data['is_buyer'] = True
        bid = best_response_scenario_buyer(price_decision_data, debug=False)
        new_B.append(bid)
        new_B_max.append(max(bid, B_max[i]))
    
    new_A = []
    new_A_min = []
    for i in range(num_sellers):
        price_decision_data['pvt_res_price'] = A_min[i]
        price_decision_data['is_buyer'] = False
        ask = best_response_scenario_seller(price_decision_data, debug=False)
        new_A.append(ask)
        new_A_min.append(min(ask, A_min[i]))

    print(f"Old Bids: {B}")
    print(f"New Bids: {new_B}")
    print(f"Old Asks: {A}")
    print(f"New Asks: {new_A}")

    B, B_max = new_B, new_B_max
    A, A_min = new_A, new_A_min

    # Prepare buyers and sellers data for market matching
    buyers = [(1, b, i, b_max, 0) for i, (b, b_max) in enumerate(zip(B, B_max))]
    sellers = [(1, a, i, a_min, 0, 0) for i, (a, a_min) in enumerate(zip(A, A_min))]

    # Perform market matching
    transactions = market_matching(buyers, sellers)

    # Process transactions
    transaction_prices_this_round = []
    for buyer_id, seller_id, quantity, price, round_num, market_advantage in transactions:
        transaction_prices.append(price)
        transaction_prices_this_round.append(price)
    
    print(f"Transactions this round: {len(transaction_prices_this_round)}")
    if transaction_prices_this_round:
        print(f"Transaction prices this round: {transaction_prices_this_round}")

    # Check for convergence in bid-ask spread
    spread = np.abs(np.mean(B) - np.mean(A))
    print(f"Current spread: {spread:.4f}")
    if spread < tolerance:
        print(f"Convergence achieved in {t+1} iterations.")
        break

    if t == max_iterations - 1:
        print(f"Max iterations ({max_iterations}) reached without convergence.")

# Plotting the convergence of average bid and ask prices
plt.figure(figsize=(10, 6))
plt.plot(avg_bid_history, label='Average Bid')
plt.plot(avg_ask_history, label='Average Ask')
plt.xlabel('Iteration')
plt.ylabel('Price')
plt.title('Convergence of Average Bid and Ask Prices')
plt.legend()
plt.grid(True)
plt.show()

# Plotting transaction prices over time
if transaction_prices:
    plt.figure(figsize=(10, 6))
    plt.plot(transaction_prices, 'o-')
    plt.xlabel('Transaction Number')
    plt.ylabel('Transaction Price')
    plt.title('Transaction Prices Over Time')
    plt.grid(True)
    plt.show()

# Plotting the demand and supply curves
price_range = np.linspace(0, 100, 100)
demand_values = [demand_curve(p, a, b) for p in price_range]
supply_values = [supply_curve(p, c, d) for p in price_range]

plt.figure(figsize=(10, 6))
plt.plot(demand_values, price_range, label='Demand Curve')
plt.plot(supply_values, price_range, label='Supply Curve')
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand and Supply Curves')
plt.legend()
plt.grid(True)
plt.show()
