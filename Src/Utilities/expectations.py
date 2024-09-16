from math import nan
import numpy as np


def get_market_demand(self, market_type):
    if self.model.step_count < 1:
        return 30, 1

    match market_type:
        case 'capital' | 'consumption' | 'labor':
            pre_transactions = getattr(self.model, f"pre_{market_type}_transactions")
            transactions = getattr(self.model, f"{market_type}_transactions")
        case _:
            raise ValueError(f"Invalid market type: {market_type}")

    latent_demand = pre_transactions[0]
    latent_price = (pre_transactions[2] + pre_transactions[3]) / 2  # Avg of buyer and seller price

    if len(transactions)>2:
        demand_realised = sum(t[2] for t in transactions)
        price_realised = sum(t[3] for t in transactions)/ len(transactions) if transactions else 0
    else:
        demand_realised, price_realised = latent_demand, latent_price

    volume_latent = latent_demand * latent_price
    volume_realised = demand_realised * price_realised


    return ((latent_demand+ demand_realised)/2, (latent_price+ price_realised)/2)


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
        last_price = current_price if current_price <0.1 else historic_prices[-1]

        expected_prices = []
        for _ in range(periods):
            # Autoregressive formula: next_price = α * last_price + (1-α) * historic_mean
            next_price = alpha * last_price + (1 - alpha) * historic_mean
            next_price = round(next_price, 2)
            expected_prices.append(next_price)
            last_price = next_price

        expected_price = np.array(expected_prices)
    else:
        historic_mean = np.mean(historic_prices)
        expected_price = np.array([historic_mean] * periods)

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
