import numpy as np
from collections import defaultdict

def update_dictionary(dictionary, dictionary_log):
    for key, value in dictionary.items():
        if key not in dictionary_log:
            dictionary_log[key] = []
        dictionary_log[key].append(value)
    return dictionary_log

def calculate_averages(dictionary, n=5):
    return {key: np.mean(value[-n:]) for key, value in dictionary.items() if value}

def dict_arithmetic(d1, d2, operation):
    result = {}
    all_keys = set(d1) | set(d2)
    
    for key in all_keys:
        v1 = d1.get(key, 0)
        v2 = d2.get(key, 0)
        try:
            result[key] = operation(v1, v2)
        except ZeroDivisionError:
            result[key] = 0  # or float('inf') or float('nan'), depending on your preference
    
    return result