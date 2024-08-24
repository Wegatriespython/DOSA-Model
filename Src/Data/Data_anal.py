import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import re

# Set the style for the plots
#plt.style.use('seaborn')

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the model and agent data."""
    model_data = pd.read_csv('model_data.csv', index_col=0)
    agent_data = pd.read_csv('agent_data.csv')

    # Convert string representations of NumPy arrays to actual lists
    for col in ['Average Market Demand', 'Total Demand']:
        model_data[col] = model_data[col].apply(lambda x: np.fromstring(re.sub('[\[\]]', '', x), sep=' ').tolist())

    return model_data, agent_data

def plot_model_variables(data: pd.DataFrame, variables: List[str], title: str, filename: str):
    """Plot selected variables from the model data."""
    plt.figure(figsize=(12, 6))
    for var in variables:
        if isinstance(data[var].iloc[0], list):
            plt.plot(data.index, [np.mean(x) for x in data[var]], label=f'Mean {var}')
        else:
            plt.plot(data.index, data[var], label=var)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
def plot_aggregate_supply_demand(agent_data: pd.DataFrame, market_type: str):
    """Plot aggregate supply and demand over time for a specific market."""
    plt.figure(figsize=(12, 6))

    supply = []
    demand = []
    steps = sorted(agent_data['Step'].unique())

    for step in steps:
        step_data = agent_data[agent_data['Step'] == step]

        if market_type == 'labor':
            supply.append(step_data[step_data['Type'] == 'Worker']['Labor'].sum())
            demand.append(step_data[step_data['Type'].isin(['Firm1', 'Firm2'])]['Labor'].sum())
        elif market_type == 'capital':
            supply.append(step_data[step_data['Type'] == 'Firm1']['Capital'].sum())
            demand.append(step_data[step_data['Type'] == 'Firm2']['Capital'].sum())
        elif market_type == 'consumption':
            supply.append(step_data[step_data['Type'] == 'Firm2']['Production'].sum())
            demand.append(step_data[step_data['Type'] == 'Worker']['Consumption'].sum())

    plt.plot(steps, supply, label='Supply', marker='o')
    plt.plot(steps, demand, label='Demand', marker='o')

    plt.title(f'{market_type.capitalize()} Market: Aggregate Supply and Demand Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Quantity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{market_type}_market_aggregate_supply_demand.png')
    plt.close()

def plot_supply_demand_curves(agent_data: pd.DataFrame, market_type: str):
    """Plot supply and demand curves for a specific market."""
    plt.figure(figsize=(12, 6))

    if market_type == 'labor':
        supply_data = agent_data[agent_data['Type'] == 'Worker'].groupby('Wage')['Labor'].sum().sort_index()
        demand_data = agent_data[agent_data['Type'].isin(['Firm1', 'Firm2'])].groupby('Wage')['Labor'].sum().sort_index()
    elif market_type == 'capital':
        supply_data = agent_data[agent_data['Type'] == 'Firm1'].groupby('Price')['Capital'].sum().sort_index()
        demand_data = agent_data[agent_data['Type'] == 'Firm2'].groupby('Price')['Capital'].sum().sort_index()
    elif market_type == 'consumption':
        supply_data = agent_data[agent_data['Type'] == 'Firm2'].groupby('Price')['Production'].sum().sort_index()
        demand_data = agent_data[agent_data['Type'] == 'Worker'].groupby('Price')['Consumption'].sum().sort_index()

    plt.plot(supply_data.values, supply_data.index, label='Supply')
    plt.plot(demand_data.values, demand_data.index, label='Demand')

    plt.title(f'{market_type.capitalize()} Market: Supply and Demand Curves')
    plt.xlabel('Quantity')
    plt.ylabel('Price/Wage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{market_type}_market_supply_demand_curves.png')
    plt.close()

def plot_firm2_data(agent_data: pd.DataFrame):
    """Plot capital, inventory, and production for Firm2 over time."""
    firm2_data = agent_data[agent_data['Type'] == 'Firm2']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    for firm in firm2_data['AgentID'].unique():
        firm_data = firm2_data[firm2_data['AgentID'] == firm]
        ax1.plot(firm_data['Step'], firm_data['Capital'], label=f'Firm {firm}')
        ax2.plot(firm_data['Step'], firm_data['Inventory'], label=f'Firm {firm}')
        ax3.plot(firm_data['Step'], firm_data['Production'], label=f'Firm {firm}')

    ax1.set_title('Firm2 Capital Over Time')
    ax1.set_ylabel('Capital')
    ax1.legend()

    ax2.set_title('Firm2 Inventory Over Time')
    ax2.set_ylabel('Inventory')
    ax2.legend()

    ax3.set_title('Firm2 Production Over Time')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Production')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('firm2_data.png')
    plt.close()

def plot_representative_agents(agent_data: pd.DataFrame):
    """Plot data for randomly sampled representative agents."""
    firm1_sample = np.random.choice(agent_data[agent_data['Type'] == 'Firm1']['AgentID'].unique(), 1, replace=False)
    firm2_sample = np.random.choice(agent_data[agent_data['Type'] == 'Firm2']['AgentID'].unique(), 2, replace=False)
    worker_sample = np.random.choice(agent_data[agent_data['Type'] == 'Worker']['AgentID'].unique(), 4, replace=False)

    samples = {'Firm1': firm1_sample, 'Firm2': firm2_sample, 'Worker': worker_sample}

    for agent_type, agent_ids in samples.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Representative {agent_type} Data')

        sample_data = agent_data[agent_data['AgentID'].isin(agent_ids)]

        if agent_type in ['Firm1', 'Firm2']:
            variables = ['Capital', 'Inventory', 'Production', 'Price']
        else:  # Worker
            variables = ['Wage', 'Skills', 'Savings', 'Consumption']

        for (i, j), var in zip([(0, 0), (0, 1), (1, 0), (1, 1)], variables):
            for agent_id in agent_ids:
                agent_data_subset = sample_data[sample_data['AgentID'] == agent_id]
                axes[i, j].plot(agent_data_subset['Step'], agent_data_subset[var], label=f'Agent {agent_id}')

            axes[i, j].set_title(f'{var} Over Time')
            axes[i, j].set_xlabel('Time Step')
            axes[i, j].set_ylabel(var)
            axes[i, j].legend()

        plt.tight_layout()
        plt.savefig(f'{agent_type.lower()}_representative_data.png')
        plt.close()

def main():
    model_data, agent_data = load_data()

    # Original plots
    plot_model_variables(model_data,
                         ['Total Labor', 'Total Capital', 'Total Goods'],
                         'Economic Aggregates Over Time',
                         'economic_aggregates.png')

    plot_model_variables(model_data,
                         ['Average Capital Price', 'Average Wage', 'Average Consumption Good Price'],
                         'Market Indicators Over Time',
                         'market_indicators.png')

    plot_model_variables(model_data,
                         ['Total Production', 'Global Productivity'],
                         'Production and Productivity Over Time',
                         'production_productivity.png')

    # New plots
    for market in ['labor', 'capital', 'consumption']:
        plot_aggregate_supply_demand(agent_data, market)
        plot_supply_demand_curves(agent_data, market)

    plot_firm2_data(agent_data)

    plot_representative_agents(agent_data)

    print("Analysis complete. Plots have been saved.")

if __name__ == "__main__":
    main()
