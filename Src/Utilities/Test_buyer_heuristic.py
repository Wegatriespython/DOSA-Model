from Strategic_adjustments import buyer_heuristic
import numpy as np
import sys
import inspect
import logging
from functools import wraps

Price_decision_data = {'is_buyer': True, 'round_num': 1, 'advantage': 'none', 'price': np.float64(1.5), 'avg_buyer_price': np.float64(2.0), 'avg_seller_price': np.float64(1.0), 'avg_seller_min_price': np.float64(0.7), 'avg_buyer_max_price': np.float64(2.46875), 'std_buyer_price': np.float64(0.0), 'std_seller_price': np.float64(0.0), 'std_buyer_max': np.float64(0.0), 'std_seller_min': np.float64(0.0), 'demand': np.float64(30.0), 'supply': np.float64(10.0), 'pvt_res_price': 1.4375, 'previous_price': 2}







if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    print(buyer_heuristic(Price_decision_data))