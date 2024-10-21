from math import isnan, nan
import numpy as np
from scipy import stats

import json
from datetime import datetime
import os

_last_update_step = -1
_market_data = []

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
            all_supply = self.model.pre_labor_transactions[1]
            print(f"all_supply: {all_supply}, market_type: {market_type}")

        case 'capital', _:
            all_supply = self.model.pre_capital_transactions[1]

        case 'consumption', _:
            all_supply = self.model.pre_consumption_transactions[1]
            print(f"all_supply: {all_supply}, market_type: {market_type}")
        case _, _:
            raise ValueError(f"Invalid market type: {market_type}")

    return round(all_supply, 2)


def get_market_stats(self, market_type):
    global _last_update_step
    print(f"Getting market stats for {market_type} at step {self.model.step_count}")
    base_result = {
        'quantity': 0, 
        'price': 0, 
        'round_num': 0, 
        'advantage': "none",
        'avg_buyer_price': 0, 
        'avg_seller_price': 0,
        'avg_buyer_max_price': 0, 
        'avg_seller_min_price': 0,
        'std_buyer_price': 0, 
        'std_seller_price': 0,
        'std_buyer_max': 0, 
        'std_seller_min': 0,
        'demand': 0,
        'supply': 0
   }

    if self.model.step_count < 1:
        
        match market_type:
            case 'capital':
                base_result.update({'quantity': 6, 'price': 3, 'round_num': 1, 'advantage': "none",   
                    'avg_buyer_max_price': 3, 'avg_seller_min_price': 3, 'avg_buyer_price': 3, 'avg_seller_price': 3,
                    'std_buyer_price': 0, 'std_seller_price': 0, 'std_buyer_max': 0, 'std_seller_min': 0, 'total_demand': 0, 'total_supply': 0})
                return base_result
                
            case 'consumption':
                base_result.update({'quantity': 30, 'price': 1, 'round_num': 1, 'advantage': "none",
                    'avg_buyer_max_price': 1, 'avg_seller_min_price': 1, 'avg_buyer_price': 1, 'avg_seller_price': 1,
                    'std_buyer_price': 0, 'std_seller_price': 0, 'std_buyer_max': 0, 'std_seller_min': 0, 'total_demand': 30, 'total_supply': 30})
                return base_result
            case 'labor':
                base_result.update({'quantity': 300, 'price': 0.0625, 'round_num': 1, 'advantage': "none",
                    'avg_buyer_max_price': 0.0625, 'avg_seller_min_price': 0.0625, 'avg_buyer_price': 0.0625, 'avg_seller_price': 0.0625,
                    'std_buyer_price': 0, 'std_seller_price': 0, 'std_buyer_max': 0, 'std_seller_min': 0, 'total_demand': 480, 'total_supply': 480})
                return base_result

    if _last_update_step != self.model.step_count:
        update_market_statistics(self)
        _last_update_step = self.model.step_count
    
    pre_transactions = getattr(self.model, f"pre_{market_type}_transactions")
    
    total_demand, total_supply, avg_buyer_price, avg_seller_price, avg_buyer_max_price, avg_seller_min_price, std_buyer_price, std_seller_price, std_buyer_max, std_seller_min = pre_transactions

    base_result.update({'demand': total_demand, 'supply': total_supply, 'avg_buyer_price': avg_buyer_price, 'avg_seller_price': avg_seller_price, 'avg_buyer_max_price': avg_buyer_max_price, 'avg_seller_min_price': avg_seller_min_price,
    'std_buyer_price': std_buyer_price, 'std_seller_price': std_seller_price, 'std_buyer_max': std_buyer_max, 'std_seller_min': std_seller_min})

    match (total_demand, total_supply, avg_buyer_price, avg_seller_price, avg_buyer_max_price, avg_seller_min_price) :
        case (0, _, _, _, _, _):
            # 0 Demand Case, returns with supply as demand
            
            base_result.update({'quantity':total_supply , 'price': avg_seller_min_price, 'round_num': 2, 'advantage': "buyer"})
            return base_result
        case (_, 0, _, _, _, _):
            # 0 Supply Case, returns with full demand at max price
            base_result.update({'quantity': total_demand, 'price': avg_buyer_max_price, 'round_num': 2, 'advantage': "seller"})
            return base_result
        

        case (_, _, buyer_price, seller_price, _, _) if buyer_price >= seller_price:
            # Round 1 Case, ideal market function  
            base_result.update({'quantity': total_demand, 'price': (buyer_price + seller_price) / 2, 'round_num': 1, 'advantage': "none"})
            return base_result

        case (demand, supply, _, seller_price, buyer_max_price, _) if demand > supply:
            # Round 2 Case, supply constrained
            match (buyer_max_price, seller_price):
                case (bid, ask) if bid >= ask:
                    # Seller is pricing correctly
                    base_result.update({
                        'quantity': supply, 
                        'price': (bid + ask) / 2, 
                        'round_num': 2, 
                        'advantage': "seller"})
                    return base_result
                case _:
                    # Seller has priced themselves out of the market
                    base_result.update({
                        'quantity': supply, 
                        'price': buyer_max_price, 
                        'round_num': 2, 
                        'advantage': "failure",
                    })
                    return base_result
        case (demand, supply, buyer_price, _, _, seller_min_price) if demand <= supply:
            # Round 2 Case, demand constrained
            match (buyer_price, seller_min_price):
                case (bid, ask) if bid >= ask:
                    # Buyer is pricing correctly
                    base_result.update({'quantity': demand, 'price': (bid + ask) / 2, 'round_num': 2, 'advantage': "buyer"})
                    return base_result
                case _:
                    # Buyer has priced themselves out of the market
                    base_result.update({'quantity': demand, 'price': seller_min_price, 'round_num': 2, 'advantage': "failure"})
                    return base_result

        case _:
            # Catch all case, failure
            base_result.update({'quantity': total_demand, 'price': avg_buyer_price, 'round_num': 2, 'advantage': "failure"})
            return base_result

def update_market_statistics(self):
    global _market_data
    print(f"Updating market statistics for step {self.model.step_count}")
    
    step_data = {
        'step': self.model.step_count,
        'markets': {}
    }

    for market_type in ['capital', 'consumption', 'labor']:
        pre_transactions = getattr(self.model, f"pre_{market_type}_transactions")
        total_demand, total_supply, avg_buyer_price, avg_seller_price, avg_buyer_max_price, avg_seller_min_price, std_buyer_price, std_seller_price, std_buyer_max, std_seller_min = pre_transactions

        # Calculate market data
        if total_demand == 0:
            clearing_quantity = (total_supply + total_demand) / 2
            clearing_price = avg_seller_min_price
            round_num = 2
            market_advantage = "buyer"
        elif total_supply == 0:
            clearing_quantity = (total_supply + total_demand) / 2
            clearing_price = avg_buyer_max_price
            round_num = 2
            market_advantage = "seller"
        elif avg_buyer_price >= avg_seller_price:
            clearing_price = (avg_buyer_price + avg_seller_price) / 2
            clearing_quantity = min(total_demand, total_supply)
            round_num = 1
            market_advantage = "none"
        elif total_demand > total_supply and avg_buyer_max_price >= avg_seller_price:
            clearing_price = (avg_buyer_max_price + avg_seller_price) / 2
            clearing_quantity = total_supply
            round_num = 2
            market_advantage = "seller"
        elif total_demand <= total_supply and avg_buyer_price >= avg_seller_min_price:
            clearing_price = (avg_buyer_price + avg_seller_min_price) / 2
            clearing_quantity = total_demand
            round_num = 2
            market_advantage = "buyer"
        else:
            clearing_quantity = total_demand
            clearing_price = avg_buyer_price
            round_num = 2
            market_advantage = "failure"

        market_data = {
            'total_demand': total_demand,
            'total_supply': total_supply,
            'avg_buyer_price': avg_buyer_price,
            'avg_seller_price': avg_seller_price,
            'avg_buyer_max_price': avg_buyer_max_price,
            'avg_seller_min_price': avg_seller_min_price,
            'std_buyer_price': std_buyer_price,
            'std_seller_price': std_seller_price,
            'std_buyer_max': std_buyer_max,
            'std_seller_min': std_seller_min,
            'clearing_quantity': clearing_quantity,
            'clearing_price': clearing_price,
            'round_num': round_num,
            'market_advantage': market_advantage
        }

        step_data['markets'][market_type] = market_data

    _market_data.append(step_data)

    # Optional: Save data to disk periodically to avoid memory issues
    if self.model.step_count == self.model.config.TIME_HORIZON - 10:
        save_market_data_to_disk(_market_data)
        _market_data = []  # Clear the in-memory data after saving

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_market_data_to_disk(data):
    # Create a datetime-based filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"market_data_{current_time}.json"
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full file path
    file_path = os.path.join(current_dir, filename)
    
    # Convert NumPy types to Python types
    converted_data = json.loads(json.dumps(data, default=numpy_to_python))
    
    # Save the data as JSON
    with open(file_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Market data saved to {file_path}")

# Function to retrieve all market data (for use in training your transformer)
def get_all_market_data():
    global _market_data
    # If you're saving to disk, you'll need to load and combine with in-memory data here
    return _market_data
