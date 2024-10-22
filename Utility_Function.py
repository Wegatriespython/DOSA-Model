import math
import numpy as np
import matplotlib.pyplot as plt
from Src.Utilities.Turbo_utility import maximize_utility

def evaluate_objective(consumption, leisure, alpha, discount_rate, periods):
    epsilon = 1e-6
    utility = sum(
        (((consumption[t] + epsilon) ** alpha) *
         (leisure[t] + epsilon) ** (1 - alpha)) * (1 / (1 + discount_rate) ** t)
        for t in range(periods)
    )
    return utility

# Constants
alpha = 0.9
discount_rate = 0.05
periods = 1
max_working_hours = 16
wage = 0.0625
price = 1


# Create a range of working hours (which equals consumption)
working_hours_range = np.linspace(0, max_working_hours, 1000)

# Calculate utility for each point
utilities = []
for working_hours in working_hours_range:
    consumption = (working_hours * wage) / price
    leisure = max_working_hours - working_hours
    utility = evaluate_objective([consumption], [leisure], alpha, discount_rate, periods)
    utilities.append(utility)

# Find the point of highest utility
max_utility_index = np.argmax(utilities)
optimal_working_hours = working_hours_range[max_utility_index]
max_utility = utilities[max_utility_index]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(working_hours_range, utilities)
plt.xlabel('Working Hours / Consumption')
plt.ylabel('Utility')
plt.title('Utility as a function of Working Hours (= Consumption)')
plt.axvline(x=optimal_working_hours, color='r', linestyle='--', label=f'Optimal: {optimal_working_hours:.2f}')
plt.legend()
plt.grid(True)
plt.show()

print(f"Maximum Utility: {max_utility}")
print(f"Optimal Working Hours / Consumption: {optimal_working_hours}")
print(f"Optimal Leisure: {max_working_hours - optimal_working_hours}")
