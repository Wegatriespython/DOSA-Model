from pandas.core.arrays.base import mode
from mesa_economy import EconomyModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_model(steps, timeout=300):
    model = EconomyModel(num_workers=30, num_firm1=2, num_firm2=5, mode='decentralised')
    for i in range(steps):
        model.step()
    return model

# Run the model
model = run_model(100)

# Get the agent data
agent_data = model.datacollector.get_agent_vars_dataframe()

# Filter for Firm1 and Firm2
firm_data = agent_data[agent_data['Type'].isin(['Firm1', 'Firm2'])]

# Reset the index to make 'Step' and 'AgentID' columns
firm_data = firm_data.reset_index()

# Print summary statistics
print("\nSummary Statistics for Firm Budgets:")
print(firm_data.groupby('AgentID')['Budget'].describe())

# Calculate the change in budget for each firm
firm_data['Budget_Change'] = firm_data.groupby('AgentID')['Budget'].diff()

# Print the firms with the largest budget increases
print("\nFirms with Largest Budget Increases:")
largest_increases = firm_data.groupby('AgentID')['Budget_Change'].max().sort_values(ascending=False).head()
print(largest_increases)

# Plot the budget over time for each firm
plt.figure(figsize=(12, 6))
for agent_id in firm_data['AgentID'].unique():
    agent_data = firm_data[firm_data['AgentID'] == agent_id]
    plt.plot(agent_data['Step'], agent_data['Budget'], label=f'Firm {agent_id}')

plt.xlabel('Step')
plt.ylabel('Budget')
plt.title('Firm Budgets Over Time')
plt.legend()
plt.show()

# Analyze correlation between budget and other variables
correlation_vars = ['Budget', 'Capital', 'Labor', 'Production', 'Price', 'Inventory']
correlation_matrix = firm_data[correlation_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Firm Variables')
plt.show()

# Analyze budget changes
plt.figure(figsize=(12, 6))
sns.boxplot(x='AgentID', y='Budget_Change', data=firm_data)
plt.title('Distribution of Budget Changes by Firm')
plt.xlabel('Firm ID')
plt.ylabel('Budget Change')
plt.show()

# Analyze relationship between production and budget
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Production', y='Budget', hue='AgentID', data=firm_data)
plt.title('Relationship between Production and Budget')
plt.xlabel('Production')
plt.ylabel('Budget')
plt.show()

print("\nAverage Budget by Firm Type:")
print(firm_data.groupby('Type')['Budget'].mean())

print("\nMedian Budget Change by Firm Type:")
print(firm_data.groupby('Type')['Budget_Change'].median())
