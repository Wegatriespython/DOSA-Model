import numpy as np
import matplotlib.pyplot as plt

def evaluate_profit(production, labor, capital, alpha, price, wage, discount_rate, periods):
    profit = sum(
        (price * production_function(labor[t], capital, alpha) - wage * labor[t] - capital) * (1 / (1 + discount_rate) ** t)
        for t in range(periods)
    )
    return profit

def production_function(labor, capital, alpha):
    epsilon = 1e-6
    return (labor + epsilon) ** alpha * (capital + epsilon) ** (1 - alpha)

# Constants
alpha = 0.9
discount_rate = 0.05
periods = 1
price = 1
wage = 0.0625
capital = 6
max_labor = 16 * 30 / 5  # Maximum labor hours

# Create a range of labor values
labor_range = np.linspace(0, max_labor, 1000)

# Calculate profit for each labor level
profits = []
for labor in labor_range:
    production = production_function(labor, capital, alpha)
    profit = evaluate_profit([production], [labor], capital, alpha, price, wage, discount_rate, periods)
    profits.append(profit)

# Find the point of highest profit
max_profit_index = np.argmax(profits)
optimal_labor = labor_range[max_profit_index]
max_profit = profits[max_profit_index]
optimal_production = production_function(optimal_labor, capital, alpha)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(labor_range, profits)
plt.xlabel('Labor')
plt.ylabel('Profit')
plt.title('Profit as a function of Labor')
plt.axvline(x=optimal_labor, color='r', linestyle='--', label=f'Optimal Labor: {optimal_labor:.2f}')
plt.legend()
plt.grid(True)
plt.show()

print(f"Maximum Profit: {max_profit}")
print(f"Optimal Labor: {optimal_labor}")
print(f"Optimal Production: {optimal_production}")
