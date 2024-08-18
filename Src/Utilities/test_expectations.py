import numpy as np
from math import isnan
from expectations import expect_demand, expect_price


def test_expect_demand():
    print("Testing expect_demand function:")

    # Test case 1: Normal case
    buyer_demand = [100, 200, 300]
    historic_demand = [150, 250, 350]
    historic_sales = [100, 200, 300]
    historic_inventory = [120, 230, 330]
    current_price = 10
    result = expect_demand(buyer_demand, historic_demand, historic_sales, historic_inventory)
    print("Test case 1 result:", result)

    # Test case 2: Periodically zero sales with fluctuations
    buyer_demand = [100, 200, 300]
    historic_demand = [150, 250, 350, 200, 300, 400, 150, 250, 350, 200, 300, 400]
    historic_sales =     [0, 50, 0, 150, 0, 200, 0, 5, 0, 100, 0, 300]
    historic_inventory = [10, 60, 200, 0, 220, 10,5, 10, 120, 0, 300,300]
    result = expect_demand(buyer_demand, historic_demand, historic_sales, historic_inventory)
    print("Test case 2 result:", result)


def test_expect_price():
    print("\nTesting expect_price function:")

    # Test case 1: Normal case
    historic_prices = [10, 11, 12, 13, 14, 15]
    current_price = 16
    result = expect_price(historic_prices, current_price)
    print("Test case 1 result:", result)

    # Test case 2: Not enough historical data
    result = expect_price([10, 11], current_price)
    print("Test case 2 result:", result)

    # Test case 3: Decreasing trend
    historic_prices = [15, 14, 13, 12, 11, 10]
    result = expect_price(historic_prices, current_price)
    print("Test case 3 result:", result)

if __name__ == "__main__":
    test_expect_demand()
    test_expect_price()
