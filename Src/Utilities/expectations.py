from math import nan
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pmdarima import auto_arima
from statsmodels.tsa.api import VAR

def get_market_demand(self, market_type):
#grab the actual demand and prices if transactions occured, else grab the pre transactions
  if self.model.step_count < 1:
    return 0
  if market_type == 'capital':
    demand = self.model.pre_capital_transactions[0]
    price = (self.model.pre_capital_transactions[2]+self.model.pre_capital_transactions[3])/2
    demand = demand
    if demand is None or price is None:
      return 0, 0
    return demand, price
  elif market_type == 'consumption':

    demand = self.model.pre_consumption_transactions[0]
    price = (self.model.pre_consumption_transactions[2]+self.model.pre_consumption_transactions[3])/2

    demand = demand
    if demand is None or price is None:
      return 0, 0
    return demand, price
  else :
    demand = self.model.pre_labor_transactions[0]
    price  = (self.model.pre_labor_transactions[2] + self.model.pre_labor_transactions[3]) / 2

    demand = demand
    if demand is None or price is None:
      return 0, 0
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



def get_expectations(demand, historic_demand, price, historic_price, periods):
   expected_prices = expect_price_ar(historic_price, price, periods)
   expected_demand = expect_demand_ar(historic_demand, demand, periods)


   return expected_prices, expected_demand


def expect_demand_ar(historic_demand, current_demand, periods=6, alpha=0.3):
    """
    Autoregressive demand expectation model centered on historic mean.

    :param historic_demand: List of historical demand
    :param current_demand: Current demand
    :param periods: Number of future periods to forecast
    :param alpha: Smoothing factor for the autoregressive component (0 < alpha < 1)
    :return: Array of expected demand
    """
    if len(historic_demand) < 5 :
        return np.full(periods, current_demand)
    else:
        historic_mean = np.mean(historic_demand)
        expected = np.full(periods, historic_mean)
        for i in range(1, min(len(historic_demand), periods) + 1):
            expected[-i] = alpha * historic_demand[-i] + (1 - alpha) * expected[-i]
        return expected

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
            next_price = round(next_price, 2)
            expected_prices.append(next_price)
            last_price = next_price

        expected_price = np.array(expected_prices)
    else:
        expected_price = np.array([current_price] * periods)

    return np.maximum(expected_price, 0)  # Ensure non-negative prices
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
