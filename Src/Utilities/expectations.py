from math import nan
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
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




def get_expectations(demand, historic_demand, price, historic_price, periods):
   expected_demand = expect_demand_ar(historic_demand,demand, periods)
   expected_price = expect_price_ar(historic_price,price, periods)
   return expected_demand, expected_price

def expect_demand_ar(historic_demand, current_demand, periods=6, alpha=0.3):
    """
    Enhanced autoregressive demand expectation model with Holt-Winters forecasting.

    :param historic_demand: List of historical demand
    :param current_demand: Current demand
    :param periods: Number of future periods to forecast
    :param alpha: Smoothing factor for the autoregressive component (0 < alpha < 1)
    :return: Array of expected demand
    """
    if len(historic_demand) < 5:
        return np.full(periods, current_demand)
    else:
        # Use Holt-Winters model for more accurate forecasting
        model = ExponentialSmoothing(historic_demand, trend='add', seasonal='add', seasonal_periods=min(len(historic_demand)//2, 12))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)

        # Combine forecast with autoregressive component
        ar_component = np.array([alpha * current_demand + (1 - alpha) * hist for hist in reversed(historic_demand[-periods:])])
        combined_forecast = 0.7 * forecast + 0.3 * ar_component[:len(forecast)]

        return np.round(combined_forecast, 2)





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
    Enhanced autoregressive price expectation model with ARIMA forecasting.

    :param historic_prices: List of historical prices
    :param current_price: Current price
    :param periods: Number of future periods to forecast
    :param alpha: Smoothing factor for the autoregressive component (0 < alpha < 1)
    :return: Array of expected prices for the specified number of periods
    """
    if len(historic_prices) > 5:
        # Use ARIMA model for more sophisticated forecasting
        model = ARIMA(historic_prices, order=(1, 1, 1))  # (p,d,q) order can be adjusted
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=periods)

        # Combine ARIMA forecast with simple autoregressive component
        historic_mean = np.mean(historic_prices)
        ar_component = []
        last_price = historic_prices[-1]
        for _ in range(periods):
            next_price = alpha * last_price + (1 - alpha) * historic_mean
            ar_component.append(next_price)
            last_price = next_price

        # Combine ARIMA forecast with autoregressive component
        combined_forecast = 0.7 * forecast + 0.3 * np.array(ar_component)
        expected_price = np.round(combined_forecast, 2)
    else:
        expected_price = np.array([current_price] * periods)

    return np.maximum(expected_price, 0)  # Ensure non-negative prices

def expect_data_ar(historic_data, current_value, periods=6, alpha=0.3):
    """
    Adaptive forecasting model for both price and demand.

    :param historic_data: List of historical data (prices or demand)
    :param current_value: Current value
    :param periods: Number of future periods to forecast
    :param alpha: Smoothing factor for the autoregressive component (0 < alpha < 1)
    :return: Array of expected values for the specified number of periods
    """
    if len(historic_data) > 10:  # Require more data for complex models
        # Use auto_arima to automatically select the best ARIMA/SARIMA model
        auto_model = auto_arima(historic_data, seasonal=True, m=12,
                                suppress_warnings=True, stepwise=True)

        # Fit the best model and forecast
        model = SARIMAX(historic_data, order=auto_model.order,
                        seasonal_order=auto_model.seasonal_order)
        fitted_model = model.fit(disp=False)
        forecast = fitted_model.forecast(steps=periods)

        # Combine with simple autoregressive component
        historic_mean = np.mean(historic_data)
        ar_component = []
        last_value = historic_data[-1]
        for _ in range(periods):
            next_value = alpha * last_value + (1 - alpha) * historic_mean
            ar_component.append(next_value)
            last_value = next_value

        # Combine forecasts
        combined_forecast = 0.7 * forecast + 0.3 * np.array(ar_component)
        expected_values = np.round(combined_forecast, 2)
    else:
        expected_values = np.array([current_value] * periods)

    return np.maximum(expected_values, 0)
