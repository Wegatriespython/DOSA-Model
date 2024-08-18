import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
model_data = pd.read_csv('model_data.csv')
agent_data = pd.read_csv('agent_data.csv')

# Set the style for the plots
sns.set(style='whitegrid')

def plot_model_variables(data, variables, title, filename):
    plt.figure(figsize=(12, 6))
    for var in variables:
        plt.plot(data.index, data[var], label=var)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to aggregate production by firm type
def aggregate_production(data, firm_type):
    return data[data['Type'] == firm_type]['Production'].sum()

# Function to aggregate demand
def aggregate_demand(data, demand_type):
    if demand_type == 'Labor':
        return data[data['Type'].isin(['Firm1', 'Firm2'])]['Labor'].sum()
    elif demand_type in ['Consumption', 'Capital']:
        return data[data['Type'] == f'Firm{2 if demand_type == "Consumption" else 1}']['Production'].sum()

# Create a new dataframe for time series data
time_series_data = pd.DataFrame()

# Aggregate production and demand over time
for step in agent_data['Step'].unique():
    step_data = agent_data[agent_data['Step'] == step]

    time_series_data.loc[step, 'Aggregate Production Firm1'] = aggregate_production(step_data, 'Firm1')
    time_series_data.loc[step, 'Aggregate Production Firm2'] = aggregate_production(step_data, 'Firm2')
    time_series_data.loc[step, 'Aggregate Demand Consumption Good'] = aggregate_demand(step_data, 'Consumption')
    time_series_data.loc[step, 'Aggregate Demand Capital Good'] = aggregate_demand(step_data, 'Capital')
    time_series_data.loc[step, 'Aggregate Demand Labor'] = aggregate_demand(step_data, 'Labor')

# Plot the new aggregate measures
plot_model_variables(time_series_data,
                     ['Aggregate Production Firm1', 'Aggregate Production Firm2'],
                     'Aggregate Production by Firm Type',
                     'aggregate_production.png')

plot_model_variables(time_series_data,
                     ['Aggregate Demand Consumption Good', 'Aggregate Demand Capital Good'],
                     'Aggregate Demand by Good Type',
                     'aggregate_demand_goods.png')

plot_model_variables(time_series_data,
                     ['Aggregate Demand Labor'],
                     'Aggregate Demand for Labor',
                     'aggregate_demand_labor.png')

# Original plots
plot_model_variables(model_data,
                     ['Total Labor', 'Total Capital', 'Total Goods'],
                     'Economic Aggregates Over Time',
                     'economic_aggregates.png')

plot_model_variables(model_data,
                     ['Average Market Demand', 'Average Capital Price', 'Average Wage'],
                     'Market Indicators Over Time',
                     'market_indicators.png')

plot_model_variables(model_data,
                     ['Total Demand', 'Total Production', 'Global Productivity'],
                     'Production and Demand Over Time',
                     'production_demand.png')

print("Analysis complete. Plots have been saved.")
