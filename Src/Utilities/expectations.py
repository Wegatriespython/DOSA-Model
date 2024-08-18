from math import nan
import numpy as np

import numpy as np

import numpy as np

def expect_demand(buyer_demand, historic_demand, historic_sales, historic_inventory, periods=6, decay_factor=0.9):
    """
    Calculate expected demand based on historical sales and demand.
    """
    buyer_demand = sum(buyer_demand)
    print(f"Input - buyer_demand: {buyer_demand}, historic_demand: {historic_demand}, historic_sales: {historic_sales}")

    # Ensure arrays are numpy arrays
    historic_demand = np.array(historic_demand)
    historic_sales = np.array(historic_sales)
    historic_inventory = np.array(historic_inventory)
    epsilon = 1e-6  # Small constant to prevent division by zero
    normalised_sales = np.minimum(1, historic_sales / np.maximum(epsilon, historic_inventory))
    print(f"normalised_sales: {normalised_sales}")

    if(len(historic_demand) - len(historic_sales)) < 0:
        historic_sales = historic_sales[-(len(historic_demand)):]
        normalised_sales = normalised_sales[-(len(historic_demand)):]
    else :
        historic_demand = historic_demand[-(len(historic_sales)):]
    # Calculate ratio array
    historic_sales = normalised_sales * historic_demand
    mask = (historic_sales != 0) & (historic_demand != 0)
    ratio_array = np.where(mask, historic_sales / historic_demand, 0)

    # Trim array to periods + 1
    ratio_array = ratio_array

    # Apply decay factor
    decayed_ratios = ratio_array * decay_factor ** np.arange(len(ratio_array)-1, -1, -1)

    # Calculate final ratio (max of decayed ratios and their mean)
    max_ratio = np.max(decayed_ratios)
    mean_ratio = np.mean(decayed_ratios)
    # Calculate expected demand
    expected_demand = min(buyer_demand * max_ratio, buyer_demand)
    print(f"{periods}, expected_demand: {expected_demand}")
    return np.full(periods, expected_demand)

def expect_price(historic_prices, current_price, periods=6):
    """
    Simplified price expectation based on historical prices.
    """
    if len(historic_prices) > 5:
        price_trend = np.mean(np.diff(historic_prices[-5:]))
        expected_price = np.array([current_price + i * price_trend for i in range(periods)])

    else:

        expected_price = np.array([current_price] * periods)

    return np.maximum(expected_price, 0)  # Ensure non-negative prices
