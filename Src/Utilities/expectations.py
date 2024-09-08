from math import nan
import numpy as np

def get_market_demand(self, market_type):
  # Rewrite to use transaction data
  #
  if self.model.step_count < 1:
    return 0
  if market_type == 'capital':
    demand = self.model.pre_capital_transactions[0]
    price = (self.model.pre_capital_transactions[2]+self.model.pre_capital_transactions[3])/2
    demand = demand / 2
    return demand, price
  elif market_type == 'consumption':
    demand = self.model.pre_consumption_transactions[0]
    price = (self.model.pre_consumption_transactions[2] + self.model.pre_consumption_transactions[3]) / 2
    demand = demand / 5
    return demand, price
  else :
    demand = self.model.pre_labor_transactions[0]
    price = (self.model.pre_labor_transactions[2] + self.model.pre_labor_transactions[3]) / 2
    demand = demand / 30
    return demand, price

def get_supply(self, market_type):
  all_supply = 0
  if market_type == 'labor':
    all_supply = self.model.pre_labor_transactions[1]
  if market_type == 'capital':

    all_supply = self.model.pre_capital_transactions[1]
  if market_type == 'consumption':
    all_supply = self.model.pre_consumption_transactions[1]
  return all_supply




def get_expectations(self ,demand, price, periods):
   expected_demand = expect_demand(demand, periods)
   expected_price = expect_price(price, periods)
   return expected_demand, expected_price


def expect_demand(demand, periods=6):
  if len(demand) >= 5:
      return np.full(periods, np.mean(demand[-5:]))
  else:
    return np.full(periods, np.mean(demand))

def expect_price(price, periods=6):
  if  len(price) >= 5:
      return np.full(periods, np.mean(price[-5:]))
  else:
    return np.full(periods, np.mean(price))

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
