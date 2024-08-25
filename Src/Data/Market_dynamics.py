import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(model_data_path, agent_data_path):
    # Load data
    model_data = pd.read_csv(model_data_path)
    agent_data = pd.read_csv(agent_data_path)

    # Convert string representations of lists to actual lists
    list_columns = ['Average Market Demand', 'Total Demand']
    for col in list_columns:
        if col in model_data.columns:
            model_data[col] = model_data[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else np.array([x]) if np.isfinite(x) else np.array([]))

    # Ensure numeric data types
    numeric_columns = ['Total Labor', 'Total Capital', 'Total Goods', 'Total Money', 'Average Capital Price',
                       'Average Wage', 'Average Inventory', 'Average Consumption Good Price', 'Total Production',
                       'Global Productivity']
    for col in numeric_columns:
        if col in model_data.columns:
            model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

    # Convert agent data to appropriate types
    numeric_agent_columns = ['Capital', 'Labor', 'Production', 'Price', 'Inventory', 'Budget', 'Productivity',
                             'Wage', 'Skills', 'Savings', 'Consumption', 'Working_Hours', 'Labor_Demand']
    for col in numeric_agent_columns:
        if col in agent_data.columns:
            agent_data[col] = pd.to_numeric(agent_data[col], errors='coerce')

    return model_data, agent_data
def plot_optimals_evolution(model_data, agent_data, market_type):
    firm_type = 'Firm1' if market_type == 'capital' else 'Firm2'
    firm_data = agent_data[agent_data['Type'] == firm_type]

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'{firm_type} Optimals Evolution for {market_type.capitalize()} Market')

    optimal_labels = ['Optimal Labor', 'Optimal Capital', 'Optimal Production', 'Optimal Price']

    for i, label in enumerate(optimal_labels):
        row = i // 2
        col = i % 2

        optimals = firm_data['Optimals'].apply(lambda x: eval(x)[i] if isinstance(x, str) else (x[i] if isinstance(x, list) else None))
        axes[row, col].plot(firm_data['Step'], optimals)
        axes[row, col].set_title(label)
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(f'{market_type}_market_optimals_evolution.png')
    plt.close()
def analyze_market(model_data, agent_data, market_type):
    plt.figure(figsize=(12, 6))

    if market_type == 'capital':
        demand = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Investment'].sum()
        supply = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Production'].sum()
        inventory = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Inventory'].sum()
        price = model_data['Average Capital Price']
        firm_type = 'Firm1'
    elif market_type == 'labor':
        demand = agent_data[agent_data['Type'] != 'Worker'].groupby('Step')['Labor_Demand'].sum()
        supply = agent_data[agent_data['Type'] == 'Worker'].groupby('Step')['Working_Hours'].sum()
        price = model_data['Average Wage']
        firm_type = 'Firm2'  # Assuming Firm2 is the main employer
    elif market_type == 'consumption':
        demand = agent_data[agent_data['Type'] == 'Worker'].groupby('Step')['Consumption'].sum()
        supply = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Production'].sum()
        inventory = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Inventory'].sum()
        price = model_data['Average Consumption Good Price']
        firm_type = 'Firm2'

    # Ensure all series have the same index
    common_index = model_data['Step']
    demand = demand.reindex(common_index)
    supply = supply.reindex(common_index)
    inventory = inventory.reindex(common_index)
    price = price.reindex(common_index)

    # Plot supply and demand
    try:
        plt.plot(common_index, demand, label='Demand')
        plt.plot(common_index, supply, label='Supply')
        plt.plot(common_index, inventory, label='Inventory')
        plt.title(f'{market_type.capitalize()} Market: Supply and Demand Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Quantity')
        plt.legend()
        plt.savefig(f'{market_type}_market_supply_demand.png')
    except Exception as e:
        print(f"Error plotting supply and demand for {market_type} market: {str(e)}")
    plt.close()

    # Price dynamics
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(common_index, price)
        plt.title(f'{market_type.capitalize()} Market: Price Dynamics')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.savefig(f'{market_type}_market_price_dynamics.png')
    except Exception as e:
        print(f"Error plotting price dynamics for {market_type} market: {str(e)}")
    plt.close()


    # Firm analysis
    firm_data = agent_data[agent_data['Type'] == firm_type]

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'{firm_type} Analysis for {market_type.capitalize()} Market')

    try:
        axes[0, 0].plot(firm_data.groupby('Step')['Capital'].mean())
        axes[0, 0].set_title('Average Capital')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Capital')

        axes[0, 1].plot(firm_data.groupby('Step')['Labor'].mean())
        axes[0, 1].set_title('Average Labor')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Labor Units')

        axes[1, 0].plot(firm_data.groupby('Step')['Budget'].mean())
        axes[1, 0].set_title('Average Budget')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Budget')

        axes[1, 1].plot(firm_data.groupby('Step')['Productivity'].mean())
        axes[1, 1].set_title('Average Productivity')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Productivity')

        plt.tight_layout()
        plt.savefig(f'{market_type}_market_firm_analysis.png')
    except Exception as e:
        print(f"Error plotting firm analysis for {market_type} market: {str(e)}")
    plt.close()

    # Calculate and print summary statistics
    print(f"\nSummary Statistics for {market_type.capitalize()} Market:")
    try:
        print(f"Average Supply-Demand Ratio: {supply.mean() / demand.mean():.2f}")
        print(f"Correlation between Price and Production: {firm_data.groupby('Step')['Production'].sum().corr(price):.2f}")

        firm_data_copy = firm_data.copy()
        firm_data_copy['Revenue'] = firm_data_copy['Production'] * firm_data_copy['Price']
        firm_data_copy['Labor_Cost'] = firm_data_copy['Labor'] * model_data.set_index('Step')['Average Wage']
        firm_data_copy['Profit'] = firm_data_copy['Revenue'] - firm_data_copy['Labor_Cost']
        average_profit_margin = firm_data_copy['Profit'].sum() / firm_data_copy['Revenue'].sum()
        print(f"Average Profit Margin: {average_profit_margin:.2f}")
    except Exception as e:
        print(f"Error calculating summary statistics for {market_type} market: {str(e)}")
    plot_optimals_evolution(model_data, agent_data, market_type)
def main():
    try:
        logging.info("Starting data loading and preprocessing...")
        model_data, agent_data = load_and_preprocess_data('model_data.csv', 'agent_data.csv')
        logging.info("Data loading and preprocessing completed successfully.")

        for market in ['capital', 'labor', 'consumption']:
            logging.info(f"Analyzing {market} market...")
            try:
                analyze_market(model_data, agent_data, market)
                logging.info(f"Analysis of {market} market completed successfully.")
            except Exception as e:
                logging.error(f"Error during analysis of {market} market: {str(e)}")

        logging.info("All analyses completed.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")

        if __name__ == "__main__":
            main()

if __name__ == "__main__":
    main()
