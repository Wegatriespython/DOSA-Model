from math import nan
import numpy as np

def expect_demand(buyer_demand, periods=6):

    return np.full(periods, buyer_demand)

def expect_price(buyer_prices, periods=6):

    return np.full(periods, np.mean(buyer_prices))

def expect_price_ar(historic_prices, current_price, periods=6, alpha=0.3):
    """
    Autoregressive price expectation model centered on historic mean.

    :param historic_prices: List of historical prices
    :param current_price: Current price
    :param periods: Number of future periods to forecast
    :param alpha: Smoothing factor for the autoregressive component (0 < alpha < 1)
    :return: Array of expected prices for the specified number of periods
    """
    if len(historic_prices) > 5:
        historic_mean = np.mean(historic_prices)
        last_price = historic_prices[-1]

        expected_prices = []
        for _ in range(periods):
            # Autoregressive formula: next_price = α * last_price + (1-α) * historic_mean
            next_price = alpha * last_price + (1 - alpha) * historic_mean
            expected_prices.append(next_price)
            last_price = next_price

        expected_price = np.array(expected_prices)
    else:
        expected_price = np.array([current_price] * periods)

    return np.maximum(expected_price, 0)  # Ensure non-negative prices
