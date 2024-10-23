import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def production_function(labor: float, capital: float, alpha: float, productivity: float) -> float:
    """Cobb-Douglas production function"""
    epsilon = 1e-6
    return productivity * (labor + epsilon) ** alpha * (capital + epsilon) ** (1 - alpha)

def calculate_profit(production: float, labor: float, capital: float, 
                    price: float, wage: float, capital_cost: float) -> float:
    """Calculate profit as revenue minus costs"""
    revenue = price * production
    labor_cost = wage * labor
    fixed_cost = capital_cost * capital
    return revenue - labor_cost - fixed_cost

def calculate_economic_profit(production: float, labor: float, capital: float, 
                            price: float, wage: float, capital_cost: float,
                            required_return: float) -> float:
    """Calculate economic profit including implicit costs"""
    revenue = price * production
    labor_cost = wage * labor
    fixed_cost = capital_cost * capital
    
    # Include opportunity cost of capital (normal return)
    opportunity_cost = capital * required_return
    
    return revenue - labor_cost - fixed_cost - opportunity_cost

def calculate_market_price(total_production: float, market_demand: float = 30.0) -> float:
    """Price falls as total production approaches/exceeds market demand"""
    return max(1.0 * (market_demand / total_production), 0.0) if total_production > 0 else 1.0

def calculate_market_wage(total_labor: float, labor_supply: float = 30.0) -> float:
    """Wage rises as total labor approaches/exceeds labor supply"""
    return max(1.0 * (total_labor / labor_supply), 1.0) if total_labor > 0 else 1.0

def calculate_competitive_profit(production: float, labor: float, capital: float,
                              n_firms: int = 5) -> float:
    """Calculate profit accounting for market competition"""
    # Total market production and labor (assuming all firms are identical)
    total_production = production * n_firms
    total_labor = labor * n_firms
    
    # Market-determined price and wage
    market_price = calculate_market_price(total_production)
    market_wage = calculate_market_wage(total_labor)
    
    revenue = production * market_price
    labor_cost = labor * market_wage
    
    return revenue - labor_cost

def find_equilibrium_price_wage(initial_price: float = 1.0, 
                              initial_wage: float = 1.0,
                              target_labor: float = 6.0,
                              target_production: float = 6.0,
                              learning_rate: float = 0.01,
                              max_iterations: int = 1000,
                              tolerance: float = 0.01) -> Dict:
    """
    Find price-wage combination that achieves target labor and production
    through iterative adjustment
    """
    price = initial_price
    wage = initial_wage
    
    history = {
        'prices': [],
        'wages': [],
        'labor': [],
        'production': [],
        'profits': []
    }
    
    for iteration in range(max_iterations):
        # Calculate production and profit at current price-wage
        labor = target_labor  # We want this to be optimal
        production = production_function(labor, CAPITAL, ALPHA, PRODUCTIVITY)
        profit = calculate_profit(production, labor, CAPITAL, price, wage, CAPITAL_COST)
        
        # Store current state
        history['prices'].append(price)
        history['wages'].append(wage)
        history['labor'].append(labor)
        history['production'].append(production)
        history['profits'].append(profit)
        
        # Check if we're at equilibrium (zero profit)
        if abs(profit) < tolerance:
            print(f"Equilibrium found after {iteration} iterations")
            break
            
        # Adjust price and wage based on profit
        if profit < 0:
            price += learning_rate  # Increase price if losing money
            wage -= learning_rate * 0.5  # Decrease wage more slowly
        else:
            price -= learning_rate * 0.5  # Decrease price more slowly
            wage += learning_rate  # Increase wage if making profit
            
        # Ensure wage doesn't go below minimum
        wage = max(wage, 0.1)
        
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Price and wage convergence
    ax1.plot(history['prices'], label='Price')
    ax1.plot(history['wages'], label='Wage')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value')
    ax1.set_title('Price and Wage Convergence')
    ax1.grid(True)
    ax1.legend()
    
    # Profit convergence
    ax2.plot(history['profits'], label='Profit')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Profit')
    ax2.set_title('Profit Convergence')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'final_price': price,
        'final_wage': wage,
        'final_profit': profit,
        'final_production': production,
        'history': history
    }

# Constants
ALPHA = 0.5  # elasticity
PRODUCTIVITY = 1.0
PRICE = 1.0
WAGE = 1.0
CAPITAL = 6.0
CAPITAL_COST = 0.0  # Assuming capital cost is sunk/fixed for this analysis
MAX_LABOR = 30
REQUIRED_RETURN = 0.15  # 15% required return on capital

# Verification calculation
test_labor = 6.0
test_production = production_function(test_labor, CAPITAL, ALPHA, PRODUCTIVITY)
test_profit = calculate_profit(test_production, test_labor, CAPITAL, PRICE, WAGE, CAPITAL_COST)
test_economic_profit = calculate_economic_profit(
    test_production, test_labor, CAPITAL, 
    PRICE, WAGE, CAPITAL_COST, REQUIRED_RETURN
)

print("\nVerification at L=6:")
print(f"Production: {test_production:.4f}")
print(f"Revenue: {(PRICE * test_production):.4f}")
print(f"Labor Cost: {(WAGE * test_labor):.4f}")
print(f"Profit: {test_profit:.4f}")
print(f"Economic Profit: {test_economic_profit:.4f}")

# Calculate over range of labor values
labor_range = np.linspace(0, MAX_LABOR, 1000)
productions = [production_function(L, CAPITAL, ALPHA, PRODUCTIVITY) for L in labor_range]
profits = [calculate_profit(Y, L, CAPITAL, PRICE, WAGE, CAPITAL_COST) 
          for Y, L in zip(productions, labor_range)]
economic_profits = [
    calculate_economic_profit(Y, L, CAPITAL, PRICE, WAGE, CAPITAL_COST, REQUIRED_RETURN)
    for Y, L in zip(productions, labor_range)
]

# Find optimal point
max_profit_index = np.argmax(profits)
optimal_labor = labor_range[max_profit_index]
optimal_production = productions[max_profit_index]
max_profit = profits[max_profit_index]

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Profit function
ax1.plot(labor_range, profits)
ax1.axvline(x=6.0, color='r', linestyle='--', label='L=6')
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.set_xlabel('Labor')
ax1.set_ylabel('Profit')
ax1.set_title('Profit as a function of Labor')
ax1.grid(True)
ax1.legend()

# Plot 2: Production function
ax2.plot(labor_range, productions)
ax2.axvline(x=6.0, color='r', linestyle='--', label='L=6')
ax2.axhline(y=6.0, color='b', linestyle='--', label='Y=6')
ax2.set_xlabel('Labor')
ax2.set_ylabel('Production')
ax2.set_title('Production as a function of Labor')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

print("\nResults:")
print(f"Optimal Labor: {optimal_labor:.4f}")
print(f"Optimal Production: {optimal_production:.4f}")
print(f"Maximum Profit: {max_profit:.4f}")

# Test the competitive equilibrium
test_points = np.linspace(1, 10, 50)
competitive_profits = []

for test_production in test_points:
    test_labor = test_production  # Given our production function
    profit = calculate_competitive_profit(test_production, test_labor, CAPITAL)
    competitive_profits.append(profit)

# Plot competitive equilibrium
plt.figure(figsize=(10, 6))
plt.plot(test_points, competitive_profits)
plt.axvline(x=6, color='r', linestyle='--', label='L=P=6')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Production/Labor per Firm')
plt.ylabel('Profit')
plt.title('Profit under Perfect Competition')
plt.grid(True)
plt.legend()
plt.show()

# Test the equilibrium finder
if __name__ == "__main__":
    # ... (previous constants remain the same)
    
    print("\nFinding equilibrium price-wage combination...")
    equilibrium = find_equilibrium_price_wage()
    
    print("\nEquilibrium Results:")
    print(f"Price: {equilibrium['final_price']:.4f}")
    print(f"Wage: {equilibrium['final_wage']:.4f}")
    print(f"Profit: {equilibrium['final_profit']:.4f}")
    print(f"Production: {equilibrium['final_production']:.4f}")
    
    # Verify that this gives us the desired labor and production levels
    test_labor = 6.0
    test_production = production_function(test_labor, CAPITAL, ALPHA, PRODUCTIVITY)
    test_profit = calculate_profit(
        test_production, test_labor, CAPITAL, 
        equilibrium['final_price'], 
        equilibrium['final_wage'], 
        CAPITAL_COST
    )
    
    print("\nVerification at L=6:")
    print(f"Production: {test_production:.4f}")
    print(f"Profit: {test_profit:.4f}")
