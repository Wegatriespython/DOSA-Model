import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_capital_market_dynamics(model_data_path, agent_data_path):
    # Load data
    model_data = pd.read_csv(model_data_path)
    agent_data = pd.read_csv(agent_data_path)

    # Filter for Firm1 data
    firm1_data = agent_data[agent_data['Type'] == 'Firm1']

    # Analyze production capacity vs demand
    plt.figure(figsize=(12, 6))
    plt.plot(model_data['Step'], model_data['Total Demand'], label='Total Demand')
    plt.plot(model_data['Step'], model_data['Total Production'], label='Total Production')
    plt.title('Total Demand vs Total Production Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Quantity')
    plt.legend()
    plt.savefig('demand_vs_production.png')
    plt.close()

    # Analyze Firm1 constraints
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Firm1 Constraints Analysis')

    # Capital constraint
    axes[0, 0].plot(firm1_data.groupby('Step')['Capital'].mean())
    axes[0, 0].set_title('Average Firm1 Capital Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Capital')

    # Labor constraint
    axes[0, 1].plot(firm1_data.groupby('Step')['Labor'].mean())
    axes[0, 1].set_title('Average Firm1 Labor Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Labor Units')

    # Budget constraint
    axes[1, 0].plot(firm1_data.groupby('Step')['Budget'].mean())
    axes[1, 0].set_title('Average Firm1 Budget Over Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Budget')

    # Productivity
    axes[1, 1].plot(firm1_data.groupby('Step')['Productivity'].mean())
    axes[1, 1].set_title('Average Firm1 Productivity Over Time')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Productivity')

    plt.tight_layout()
    plt.savefig("firm1_constraints.png")
    plt.close()

    # Analyze price dynamics
    plt.figure(figsize=(12, 6))
    plt.plot(model_data['Step'], model_data['Average Capital Price'], label='Average Capital Price')
    plt.plot(model_data['Step'], model_data['Average Wage'], label='Average Wage')
    plt.title('Price Dynamics Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Price/Wage')
    plt.legend()
    plt.savefig('price_dynamics.png')
    plt.close()

    # Analyze inventory levels
    plt.figure(figsize=(12, 6))
    plt.plot(firm1_data.groupby('Step')['Inventory'].mean())
    plt.title('Average Firm1 Inventory Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Inventory')
    plt.savefig('inventory_levels.png')
    plt.close()

    # Calculate and print some summary statistics
    print("Average Firm1 Production Capacity Utilization:")
    print(firm1_data.groupby('Step').apply(lambda x: x['Production'].sum() / x['Capital'].sum()).mean())

    print("\nCorrelation between Capital Price and Production:")
    print(pd.concat([model_data['Average Capital Price'], firm1_data.groupby('Step')['Production'].sum()], axis=1).corr().iloc[0, 1])

# Calculate average profit margin for Firm1
    print("\nAverage Profit Margin for Firm1:")
    firm1_data_copy = firm1_data.copy()
    firm1_data_copy['Profit'] = (
        firm1_data_copy['Production'] * firm1_data_copy['Price'] -
        firm1_data_copy['Labor'] * model_data.set_index('Step')['Average Wage']
    )
    average_profit_margin = (
        firm1_data_copy['Profit'].mean() /
        (firm1_data_copy['Production'] * firm1_data_copy['Price']).mean()
    )
    print(average_profit_margin)

if __name__ == "__main__":
    analyze_capital_market_dynamics('model_data.csv', 'agent_data.csv')
