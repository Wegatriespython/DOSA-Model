import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def load_and_preprocess_data(model_data_path, agent_data_path):
    # Load data
    model_data = pd.read_csv(model_data_path)
    agent_data = pd.read_csv(agent_data_path)

    # Model data preprocessing
    model_numeric_columns = [
        'Total Labor', 'Total Capital', 'Total Goods', 'Total Money',
        'Average Market Demand', 'Average Capital Price',
        'Average_Consumption_Demand', 'Average_Consumption_Demand_Expected',
        'Average_Consumption_expected_price', 'Average Wage',
        'Average Inventory', 'Average Consumption Good Price',
        'Total Demand', 'Total Supply', 'Total Production',
        'Global Productivity'
    ]

    for col in model_numeric_columns:
        if col in model_data.columns:
            model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

    # Convert string representations of lists to actual lists
    list_columns = ['Average Market Demand', 'Total Demand']
    for col in list_columns:
        if col in model_data.columns:
            model_data[col] = model_data[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else np.array([x]) if np.isfinite(x) else np.array([]))

    # Agent data preprocessing
    agent_numeric_columns = [
        'Capital', 'Labor', 'Working_Hours', 'Labor_Demand', 'Production',
        'Price', 'Inventory', 'Budget', 'Productivity', 'Wage', 'Skills',
        'Savings', 'Consumption'
    ]

    for col in agent_numeric_columns:
        if col in agent_data.columns:
            agent_data[col] = pd.to_numeric(agent_data[col], errors='coerce')

    def safe_eval(x):
        if isinstance(x, str):
            try:
                return eval(x, {'array': np.array, 'np': np})
            except:
                return x
        return x

    if 'Expectations' in agent_data.columns:
        agent_data['Expectations'] = agent_data['Expectations'].apply(safe_eval)

    if 'Optimals' in agent_data.columns:
        agent_data['Optimals'] = agent_data['Optimals'].apply(safe_eval)
    return model_data, agent_data
# Load your data
model_data, agent_data = load_and_preprocess_data('model_data.csv', 'agent_data.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Economic Model Dashboard'),

    dcc.Tabs([
        dcc.Tab(label='Markets', children=[
            dcc.Dropdown(
                id='market-dropdown',
                options=[
                    {'label': 'Capital Market', 'value': 'capital'},
                    {'label': 'Labor Market', 'value': 'labor'},
                    {'label': 'Consumption Market', 'value': 'consumption'}
                ],
                value='capital'
            ),
            dcc.Checklist(
                id='market-variables-checklist',
                options=[
                    {'label': 'Supply', 'value': 'supply'},
                    {'label': 'Demand', 'value': 'demand'},
                    {'label': 'Inventory', 'value': 'inventory'},
                    {'label': 'Price', 'value': 'price'}
                ],
                value=['supply', 'demand']
            ),
            dcc.Graph(id='market-graph')
        ]),
        dcc.Tab(label='Firms', children=[
            dcc.Dropdown(
                id='firm-type-dropdown',
                options=[
                    {'label': 'Firm1', 'value': 'Firm1'},
                    {'label': 'Firm2', 'value': 'Firm2'}
                ],
                value='Firm1'
            ),
            dcc.Dropdown(
                id='firm-graph-type',
                options=[
                    {'label': 'Performance Metrics', 'value': 'performance'},
                    {'label': 'Optimals', 'value': 'optimals'}
                ],
                value='performance'
            ),
            dcc.Graph(id='firm-graph')
        ]),
        dcc.Tab(label='Workers', children=[
            dcc.Graph(id='worker-graph')
        ])
    ])
])

@app.callback(
    Output('market-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('market-variables-checklist', 'value')]
)
def update_market_graph(selected_market, selected_variables):
    if selected_market == 'capital':
        demand = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Investment'].sum()
        supply = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Production'].sum()
        inventory = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Inventory'].sum()
        price = model_data['Average Capital Price']
    elif selected_market == 'labor':
        demand = agent_data[agent_data['Type'].isin(['Firm1', 'Firm2'])].groupby('Step')['Labor_Demand'].sum()
        supply = agent_data[agent_data['Type'] == 'Worker'].groupby('Step')['Working_Hours'].sum()
        inventory = pd.Series(0, index=demand.index)  # No inventory for labor
        price = model_data['Average Wage']
    else:  # consumption
        demand = agent_data[agent_data['Type'] == 'Worker'].groupby('Step')['Consumption'].sum()
        supply = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Production'].sum()
        inventory = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Inventory'].sum()
        price = model_data['Average Consumption Good Price']

    common_index = model_data['Step']
    figure = go.Figure()

    if 'supply' in selected_variables:
        figure.add_trace(go.Scatter(x=common_index, y=supply.reindex(common_index, fill_value=0), mode='lines', name='Supply'))
    if 'demand' in selected_variables:
        figure.add_trace(go.Scatter(x=common_index, y=demand.reindex(common_index, fill_value=0), mode='lines', name='Demand'))
    if 'inventory' in selected_variables:
        figure.add_trace(go.Scatter(x=common_index, y=inventory.reindex(common_index, fill_value=0), mode='lines', name='Inventory'))
    if 'price' in selected_variables:
        figure.add_trace(go.Scatter(x=common_index, y=price.reindex(common_index, fill_value=0), mode='lines', name='Price', yaxis='y2'))
        figure.update_layout(yaxis2=dict(title='Price', overlaying='y', side='right'))

    figure.update_layout(title=f'{selected_market.capitalize()} Market',
                         xaxis_title='Time Step', yaxis_title='Quantity')
    return figure

@app.callback(
    Output('firm-graph', 'figure'),
    [Input('firm-type-dropdown', 'value'),
     Input('firm-graph-type', 'value')]
)
def update_firm_graph(firm_type, graph_type):
    firm_data = agent_data[agent_data['Type'] == firm_type]

    if graph_type == 'performance':
        figure = go.Figure()
        for variable in ['Budget', 'Production', 'Price', 'Inventory', 'Capital']:
            data = firm_data.groupby('Step')[variable].mean()
            figure.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name=variable))
        figure.update_layout(title=f'{firm_type} Performance Metrics',
                             xaxis_title='Time Step', yaxis_title='Value')
    if graph_type =='optimals':  # optimals
        figure = go.Figure()
        optimal_data = firm_data[firm_data['Optimals'].notna()]
        for i, label in enumerate(['Optimal Labor', 'Optimal Capital', 'Optimal Production', 'Optimal Price']):
            y = optimal_data['Optimals'].apply(lambda opt: opt[i] if isinstance(opt, list) and len(opt) > i else None)
            figure.add_trace(go.Scatter(x=optimal_data['Step'], y=y, mode='lines', name=label))
        figure.update_layout(title=f'{firm_type} Optimals',
                             xaxis_title='Time Step', yaxis_title='Value')
    else:  # expectations
        figure = go.Figure()
        Expectations_data = firm_data[firm_data['Expectations'].notna()]
        for i, label in enumerate(['Expectations Demand', 'Expectations Price']):
            y = Expectations_data['Expectations'].apply(lambda opt: opt[i] if isinstance(opt, list) and len(opt) > i else None)
            figure.add_trace(go.Scatter(x=Expectations_data['Step'], y=y, mode='lines', name=label))
        figure.update_layout(title=f'{firm_type} Expectations',
                             xaxis_title='Time Step', yaxis_title='Value')
    return figure

@app.callback(
    Output('worker-graph', 'figure'),
    [Input('market-dropdown', 'value')]  # This input is not used but required by Dash
)
def update_worker_graph(_):
    worker_data = agent_data[agent_data['Type'] == 'Worker']
    avg_savings = worker_data.groupby('Step')['Savings'].mean()
    avg_working_hours = worker_data.groupby('Step')['Working_Hours'].mean()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=avg_savings.index, y=avg_savings.values, mode='lines', name='Average Savings'))
    figure.add_trace(go.Scatter(x=avg_working_hours.index, y=avg_working_hours.values, mode='lines', name='Average Working Hours', yaxis='y2'))

    figure.update_layout(title='Worker Metrics',
                         xaxis_title='Time Step',
                         yaxis_title='Savings',
                         yaxis2=dict(title='Working Hours', overlaying='y', side='right'))
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
