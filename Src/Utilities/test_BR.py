import pytest
import numpy as np
from Strategic_adjustments import best_response_exact

#def calculate_expected_utility(price, data):
#    temp_data = data.copy()
#    temp_data['previous_price'] = price
#    temp_data['pvt_res_price'] = price
#    return -best_response_exact(temp_data)

"""def simulate_market(initial_conditions, num_rounds=20):
    market_data = initial_conditions.copy()
    price_history = []
    trade_history = []
    
    for round in range(1, num_rounds + 1):
        market_data['round_num'] = round
        buyer_price = best_response_exact({**market_data, 'is_buyer': True})
        seller_price = best_response_exact({**market_data, 'is_buyer': False})
        
        price_history.append((buyer_price, seller_price))
        
        # Determine if trade occurs
        if round == 1:
            trade_occurs = buyer_price > seller_price
        else:
            if market_data['expected_demand'] > market_data['expected_supply']:
                trade_occurs = buyer_price > market_data['seller_min_price']
            else:
                trade_occurs = market_data['buyer_max_price'] > seller_price
        
        trade_history.append(trade_occurs)
        
        # Update market conditions
        market_data['avg_buyer_price'] = buyer_price
        market_data['avg_seller_price'] = seller_price
        market_data['previous_price'] = (buyer_price + seller_price) / 2
        
        # Update buyer reservation price (with volatility)
        market_data['buyer_max_price'] = np.clip(
            market_data['buyer_max_price'] * np.random.uniform(0.9, 1.1),
            0.08, 0.2
        )
    
    return price_history, trade_history

# Commented out other tests


@pytest.mark.parametrize("scenario", [
    {
        "name": "Balanced market",
        "expected_demand": 100,
        "expected_supply": 100,
        "initial_price": 0.1,
    },
    {
        "name": "High demand market",
        "expected_demand": 150,
        "expected_supply": 100,
        "initial_price": 0.1,
    },
    {
        "name": "Low demand market",
        "expected_demand": 50,
        "expected_supply": 100,
        "initial_price": 0.1,
    },
])
def test_market_clearing(scenario):
    initial_conditions = {
        'round_num': 1,
        'avg_buyer_price': scenario['initial_price'],
        'avg_seller_price': scenario['initial_price'],
        'std_buyer_price': 0.01,
        'std_seller_price': 0.01,
        'pvt_res_price': scenario['initial_price'],
        'expected_demand': scenario['expected_demand'],
        'expected_supply': scenario['expected_supply'],
        'previous_price': scenario['initial_price'],
        'buyer_max_price': 0.15,  # Initial buyer reservation price
        'seller_min_price': 0.0625  # Seller reservation price
    }
    
    price_history, trade_history = simulate_market(initial_conditions)
    
    print(f"\nScenario: {scenario['name']}")
    print("Price History:")
    for i, (bp, sp) in enumerate(price_history):
        print(f"Round {i+1}: Buyer: {bp:.4f}, Seller: {sp:.4f}, Trade: {trade_history[i]}")
    
    # Check if market clears eventually
    assert any(trade_history), "No trades occurred during the simulation"
    
    # Check if prices converge
    final_buyer_price, final_seller_price = price_history[-1]
    price_gap = abs(final_buyer_price - final_seller_price)
    assert price_gap < 0.01, f"Prices did not converge sufficiently. Final gap: {price_gap:.4f}"
    
    # Check if final price is between seller min and initial buyer max
    assert 0.0625 <= final_buyer_price <= 0.2, f"Final buyer price {final_buyer_price:.4f} is out of expected range"
    assert 0.0625 <= final_seller_price <= 0.2, f"Final seller price {final_seller_price:.4f} is out of expected range"
    
    # Check for price stability in later rounds
    late_price_changes = [abs(price_history[i][0] - price_history[i-1][0]) +
                          abs(price_history[i][1] - price_history[i-1][1])
                          for i in range(-5, -1)]
    assert max(late_price_changes) < 0.01, "Prices are not stabilizing in later rounds"

def test_path_dependency():
    base_conditions = {
        'round_num': 1,
        'avg_buyer_price': 0.1,
        'avg_seller_price': 0.1,
        'std_buyer_price': 0.01,
        'std_seller_price': 0.01,
        'pvt_res_price': 0.1,
        'expected_demand': 100,
        'expected_supply': 100,
        'previous_price': 0.1,
        'buyer_max_price': 0.15,
        'seller_min_price': 0.0625
    }
    
    # Simulate with different starting prices
    history_low, _ = simulate_market({**base_conditions, 'previous_price': 0.05})
    history_high, _ = simulate_market({**base_conditions, 'previous_price': 0.15})
    
    final_price_low = (history_low[-1][0] + history_low[-1][1]) / 2
    final_price_high = (history_high[-1][0] + history_high[-1][1]) / 2
    
    print(f"\nFinal price (starting low): {final_price_low:.4f}")
    print(f"Final price (starting high): {final_price_high:.4f}")
    
    # Assert that final prices are close, regardless of starting point
    assert abs(final_price_low - final_price_high) < 0.01, "Path dependency detected"""

"""def test_seller_price_stuck_at_minimum():
    initial_conditions = {
        'round_num': 1,
        'avg_buyer_price': 0.08,
        'avg_seller_price': 0.0625,
        'std_buyer_price': 0.01,
        'std_seller_price': 0.01,
        'pvt_res_price': 0.0625,
        'expected_demand': 800,
        'expected_supply': 240,
        'previous_price': 0.0625,
        'buyer_max_price': 0.09,
        'seller_min_price': 0.0625
    }
    
    price_history, trade_history = simulate_market(initial_conditions, num_rounds=30)
    
    print("\nScenario: Seller price stuck at minimum")
    print("Price History:")
    for i, (bp, sp) in enumerate(price_history):
        print(f"Round {i+1}: Buyer: {bp:.4f}, Seller: {sp:.4f}, Trade: {trade_history[i]}")
    
    # Check if seller price moves above the minimum
    seller_prices = [sp for _, sp in price_history]
    assert max(seller_prices) > 0.0625, "Seller price never moved above the minimum"
    
    # Check if trades occur
    assert any(trade_history), "No trades occurred during the simulation"
    
    # Check if final prices converge to a higher level
    final_buyer_price, final_seller_price = price_history[-1]
    assert final_seller_price > 0.0625, f"Final seller price {final_seller_price:.4f} did not increase above minimum"
    assert final_buyer_price > 0.08, f"Final buyer price {final_buyer_price:.4f} did not increase"
"""
def test_seller_response_to_high_demand():
    # Set up the exact conditions described
    price_decision_data = {
        'is_buyer': False,  # We're testing the seller's response
        'round_num': 1,
        'avg_buyer_price': 0.08,
        'avg_seller_price': 0.0625,
        'std_buyer_price': 0.01,
        'std_seller_price': 0.01,
        'pvt_res_price': 0.0625,
        'expected_demand': 800,
        'expected_supply': 240,
        'previous_price': 0.0625,
        'buyer_max_price': 0.09,
        'seller_min_price': 0.0625
    }
    
    # Get the seller's response
    seller_price = best_response_exact(price_decision_data)
    
    print(f"\nTesting seller response to high demand")
    print(f"Input conditions:")
    for key, value in price_decision_data.items():
        print(f"{key}: {value}")
    print(f"\
    Seller's response: {seller_price:.4f}")

if __name__ == "__main__":
    test_seller_response_to_high_demand()
