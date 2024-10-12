from math import isnan, nan
import numpy as np
from scipy import stats

def get_market_demand_simple(self, market_type):

    if self.model.step_count < 1:
        match market_type:
            case 'capital':
                return 6, 3, 6
            case 'consumption':
                return 30, 1, 6
            case 'labor':
                return  300, 0.0625, 6

    match market_type:
        case 'capital' | 'consumption' | 'labor':
            pre_transactions = getattr(self.model, f"pre_{market_type}_transactions")
            transactions = getattr(self.model, f"{market_type}_transactions")
        case _:
            raise ValueError(f"Invalid market type: {market_type}")

    latent_demand = pre_transactions[0]
    latent_price = (pre_transactions[2] + pre_transactions[3]) / 2  # Avg of buyer and seller price

    if len(transactions)>2 and market_type == 'consumption': # Irregular demand in captial markets causing issues.
        demand_realised = sum(t[2] for t in transactions)
        price_realised = sum(t[3] for t in transactions)/ len(transactions) if transactions else 0
    else:
        demand_realised, price_realised = latent_demand, latent_price


    demand = round(latent_demand ,2)
    price = round(latent_price,2)
    if isnan(demand) or isnan(price):
      print('Error', latent_demand, latent_price, demand_realised, price_realised)
      breakpoint()

    rational_expectations = get_market_demand_rational(self, market_type)
    return demand, price, 6

## Pre-{market}_transactions is cleared before we access it.

def get_supply(self, market_type):
    all_supply = 0
    
    match market_type, self.model.step_count:
        case _, 0:
            match market_type:
                case 'labor':
                    all_supply = 300
                case 'capital':
                    all_supply = 6
                case 'consumption':
                    all_supply = 25
        case 'labor', _:
            all_supply = self.model.get_total_labor_supply()
            print(f"all_supply: {all_supply}, market_type: {market_type}")

        case 'capital', _:
            all_supply = self.model.get_total_capital_supply()

        case 'consumption', _:
            all_supply = self.model.get_total_consumption_supply()
            print(f"all_supply: {all_supply}, market_type: {market_type}")
        case _, _:
            raise ValueError(f"Invalid market type: {market_type}")

    return round(all_supply, 2)


def get_market_demand(self, market_type):
    if self.model.step_count < 1:
        # Return initial values for the first step, same as get_market_demand
        match market_type:
            case 'capital':
                return 6, 3, 1, "none", 1  # quantity, price, round, market_advantage
            case 'consumption':
                return 30, 1, 1, "none", 1
            case 'labor':
                return 300, 0.0625, 1, "none", 1

    # Get pre-transaction data
    pre_transactions = getattr(self.model, f"pre_{market_type}_transactions")
    
    # Extract aggregate buyer and seller data
    total_demand = pre_transactions[0]
    avg_buyer_price = pre_transactions[2]
    avg_buyer_max_price = pre_transactions[4]
    total_supply = pre_transactions[1]
    avg_seller_price = pre_transactions[3]
    avg_seller_min_price = pre_transactions[5]

    # Round 1: Check if market clears based on average prices
    if avg_buyer_price >= avg_seller_price:
        clearing_price = (avg_buyer_price + avg_seller_price) / 2
        clearing_quantity = total_demand
        return clearing_quantity, clearing_price, 1, "none", avg_buyer_max_price

    # Round 2: Determine market advantage and adjust prices
    if total_demand > total_supply:
        # Sellers Advantage
        if avg_buyer_max_price >= avg_seller_price:
            clearing_price = (avg_buyer_max_price + avg_seller_price) / 2
            clearing_quantity = total_demand
            return clearing_quantity, clearing_price, 2, "seller", avg_buyer_max_price
    else:
        # Buyers Advantage
        if avg_buyer_price >= avg_seller_min_price:
            clearing_price = (avg_buyer_price + avg_seller_min_price) / 2
            clearing_quantity = total_demand
            return clearing_quantity, clearing_price, 2, "buyer", avg_buyer_max_price

    # If no clearing in either round, return theoretical equilibrium
    return total_demand, avg_buyer_price, 2, "failure", avg_buyer_max_price

