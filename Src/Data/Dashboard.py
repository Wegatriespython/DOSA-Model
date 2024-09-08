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
        'Savings', 'Consumption', "Wages_Firm", "Sales"
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
                    {'label': 'Price', 'value': 'price'},
                    {'label': 'Sales', 'value': 'sales'},
                    {'label': 'Wages', 'value': 'wage'},
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
                    {'label': 'Expectations', 'value': 'expectations'},
                    {'label': 'Optimals', 'value': 'optimals'}
                ],
                value='performance'
            ),
            dcc.Graph(id='firm-graph')
        ]),
        dcc.Tab(label='Workers', children=[
            dcc.Graph(id='worker-graph')
        ]),
        dcc.Tab(label='Aggregate Transactions', children=[
            dcc.Dropdown(
                id='transaction-market-dropdown',
                options=[
                    {'label': 'Capital Market', 'value': 'capital'},
                    {'label': 'Labor Market', 'value': 'labor'},
                    {'label': 'Consumption Market', 'value': 'consumption'}
                ],
                value='capital'
            ),
            dcc.Graph(id='transaction-graph'),
            dcc.Graph(id='pre-transaction-graph'),
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
        sales = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Sales'].sum()
        wage = agent_data[agent_data['Type'] == 'Firm1'].groupby('Step')['Wages_Firm'].mean()
    elif selected_market == 'labor':
        demand = agent_data[agent_data['Type'].isin(['Firm1', 'Firm2'])].groupby('Step')['Labor_Demand'].sum()
        supply = agent_data[agent_data['Type'] == 'Worker'].groupby('Step')['Working_Hours'].sum()
        inventory = pd.Series(0, index=demand.index)  # No inventory for labor
        price = model_data['Average Wage']
        sales = pd.Series(0, index=demand.index)  # No sales for labor
        wage = model_data['Average Wage']
    else:  # consumption
        demand = agent_data[agent_data['Type'] == 'Worker'].groupby('Step')['desired_consumption'].sum()
        supply = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Production'].sum()
        inventory = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Inventory'].sum()
        price = model_data['Average Consumption Good Price']
        sales = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Sales'].sum()
        wage = agent_data[agent_data['Type'] == 'Firm2'].groupby('Step')['Per_Worker_Income'].mean()

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
    if 'sales' in selected_variables:
        figure.add_trace(go.Scatter(x=common_index, y=sales.reindex(common_index, fill_value=0), mode='lines', name='Sales'))
    if 'wage' in selected_variables:
        figure.add_trace(go.Scatter(x=common_index, y=wage.reindex(common_index, fill_value=0), mode='lines', name='Wage', yaxis='y2'))

    figure.update_layout(title=f'{selected_market.capitalize()} Market',
                         xaxis_title='Time Step', yaxis_title='Quantity',
                         yaxis2=dict(title='Price/Wage', overlaying='y', side='right'))
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
        for variable in ['Budget', 'Production', 'Price', 'Inventory', 'Capital', 'Sales']:
            data = firm_data.groupby('Step')[variable].mean()
            figure.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name=variable))
        figure.update_layout(title=f'{firm_type} Performance Metrics',
                             xaxis_title='Time Step', yaxis_title='Value')
    elif graph_type == 'optimals':
        figure = go.Figure()
        optimal_data = firm_data[firm_data['Optimals'].notna()]

        def parse_numpy_array_string(s):
            try:
                # Remove brackets and split by spaces
                values = s.strip('[]').split()
                # Convert to float and return as a list
                return [float(x) for x in values]
            except:
                return None

        # Convert string representation of NumPy array to list
        optimal_data['Optimals'] = optimal_data['Optimals'].apply(parse_numpy_array_string)

        for i, label in enumerate(['Optimal Labor', 'Optimal Capital', 'Optimal Production', 'Optimal Inventory', 'Optimal Sales']):
            y = optimal_data['Optimals'].apply(lambda opt: opt[i] if isinstance(opt, list) and len(opt) > i else None)
            figure.add_trace(go.Scatter(x=optimal_data['Step'], y=y, mode='lines', name=label))

        figure.update_layout(title=f'{firm_type} Optimals',
            xaxis_title='Time Step', yaxis_title='Value')
    elif graph_type == 'expectations':
        figure = go.Figure()
        expectations_data = firm_data[firm_data['Expectations'].notna()]
        for i, label in enumerate(['Expected Demand', 'Expected Price']):
            y = expectations_data['Expectations'].apply(lambda exp: exp[i] if isinstance(exp, list) and len(exp) > i else None)
            figure.add_trace(go.Scatter(x=expectations_data['Step'], y=y, mode='lines', name=label))
        figure.update_layout(title=f'{firm_type} Expectations',
                             xaxis_title='Time Step', yaxis_title='Value')
    else:
        figure = go.Figure()  # Empty figure if no valid graph_type is selected

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
@app.callback(
    [Output('pre-transaction-graph', 'figure'),
        Output('transaction-graph', 'figure')],
    [Input('transaction-market-dropdown', 'value')]
)
def update_transaction_graph(selected_market):
    pre_transaction_fig = go.Figure()
    transaction_fig = go.Figure()

    # Pre-transaction data
    pre_transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y=model_data[f'{selected_market}_raw_demand'], mode='lines', name='Demand'))
    pre_transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y=model_data[f'{selected_market}_raw_supply'], mode='lines', name='Supply'))
    pre_transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y=model_data[f'{selected_market}_raw_buyer_price'], mode='lines', name='Buyer Price', yaxis='y2'))
    pre_transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y=model_data[f'{selected_market}_raw_seller_price'], mode='lines', name='Seller Price', yaxis='y2'))
    pre_transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y = model_data[f'{selected_market}_raw_buyer_max'], mode ='lines', name ='Buyer_Max', yaxis ='y2'))
    pre_transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y = model_data[f'{selected_market}_raw_seller_min'], mode = 'lines', name ='Seller_Min', yaxis ='y2'))


    pre_transaction_fig.update_layout(
        title=f'{selected_market.capitalize()} Market - Pre-Transaction Data',
        xaxis_title='Time Step',
        yaxis_title='Quantity',
        yaxis2=dict(title='Price', overlaying='y', side='right')
    )

    # Transaction data
    transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y=model_data[f'{selected_market}_Market_Quantity'], mode='lines', name='Quantity'))
    transaction_fig.add_trace(go.Scatter(x=model_data['Step'], y=model_data[f'{selected_market}_Market_Price'], mode='lines', name='Price', yaxis='y2'))

    transaction_fig.update_layout(
        title=f'{selected_market.capitalize()} Market - Transaction Data',
        xaxis_title='Time Step',
        yaxis_title='Quantity',
        yaxis2=dict(title='Price', overlaying='y', side='right')
    )

    return pre_transaction_fig, transaction_fig
if __name__ == '__main__':
    app.run_server(debug=True)
