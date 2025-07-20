import dash
from dash import html, dcc, Input, Output, State, callback, ctx, dash_table, ALL
import dash_bootstrap_components as dbc
import pandas as pd


def get_initial_table_data(states):
    initial_table_data = [{'state': state, 'min': 0, 'max': 100} for state in states]
    return initial_table_data

def get_initial_intensity_data(intensity_matrix_data,states):
    intensity_matrix = pd.DataFrame(data=list(map(lambda d: {k: v for k, v in d.items() if k != 'State'}, intensity_matrix_data)),index=states,columns=states)
    all_transitions = [
        (from_state, to_state)
        for from_state in states
        for to_state in states
        if from_state != to_state and intensity_matrix.loc[from_state, to_state] != 0
    ]
    initial_table_data = [{'from_state': from_state,'to_state':to_state, 'min': 0.01, 'max': 0.15} for (from_state,to_state) in all_transitions]
    return initial_table_data

intervals_table = dash_table.DataTable(
    id='intervals-table',
    columns=[
        {'name': 'State', 'id': 'state', 'type': 'text', 'editable': False},
        {'name': 'Min % Document Count', 'id': 'min', 'type': 'numeric', 'editable': True},
        {'name': 'Max % Document Count', 'id': 'max', 'type': 'numeric', 'editable': True}
    ],
    data=[],
    editable=True,
    row_deletable=True,
)

intensities_table = dash_table.DataTable(
    id='intensities-table',
    columns=[
        {'name': 'From State', 'id': 'from_state', 'presentation': 'dropdown', 'editable': True},
        {'name': 'To State', 'id': 'to_state', 'presentation': 'dropdown', 'editable': True},
        {'name': 'Min intensity', 'id': 'min', 'type': 'numeric', 'editable': True},
        {'name': 'Max intensity', 'id': 'max', 'type': 'numeric', 'editable': True}
    ],
    data=[],
    editable=True,
    row_deletable=True,
)

# Layout for step 2
layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4([
                "States Dictionary",
                html.I(className="fas fa-question-circle", id="tooltip-target-states-dict-step2", style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
    "This tooltip provides additional information about the intensity matrix.",
    target="tooltip-target-states-dict-step2",
    placement="right",
            )])]),
    dbc.Row([
        dbc.Col([
            html.Div(id='states-dict-step2-div',style={'max-height':'15vh','overflow-y':'auto','margin-bottom':'15px'}),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Optimization Criteria",
                html.I(className="fas fa-question-circle", id="tooltip-target-optimization-criteria",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-optimization-criteria",
                placement="right",
            )])]),
    dbc.Row(dbc.Col(html.Div([
        html.H5("1. Maximization of revenue from documents in productive states"),
                html.Div([
                    html.Img(src="/assets/fun1.jpg", style={'display': 'block', 'margin': '0 auto'})
                ]),
                html.Br(),
    ]))),
    dbc.Row(dbc.Col(html.Div([
        html.H5("2. Minimization of loss risk from document in non-productive states"),
                html.Div([
                    html.Img(src="/assets/fun2.jpg", style={'display': 'block', 'margin': '0 auto'})
                ]),
                html.Br(),
    ]))),
    dbc.Row(dbc.Col(html.Div([
        html.H5("3. Minimization of revenue loss risk from documents in productive states"),
                html.Div([
                    html.Img(src="/assets/fun3.jpg", style={'display': 'block', 'margin': '0 auto'})
                ]),
                html.Br(),
    ]))),
    dbc.Row(dbc.Col(html.Div([
        html.H5("4. Risk-return weighted assessment"),
                html.Div([
                    html.Img(src="/assets/fun4.jpg", style={'display': 'block', 'margin': '0 auto'})
                ]),
                html.Br(),
    ]))),
    dbc.Row(dbc.Col(html.Div([
        html.H5("5. Risk-return weighted assessment"),
                html.Div([
                    html.Img(src="/assets/fun5.jpg", style={'display': 'block', 'margin': '0 auto'})
                ]),
                html.Br(),
    ]))),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Productive and Unproductive States",
                html.I(className="fas fa-question-circle", id="tooltip-target-productive-states",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-productive-states",
                placement="right",
            )])],style={'margin-top':'30px'}),
    dbc.Row([
                dbc.Col([
                    html.Label("Productive States"),
                    dcc.Dropdown(
                        id='productive-states-dropdown',
                        #options=[{'label': state, 'value': state} for state in states],
                        value=[],
                        multi=True,
                        persistence=True,
                        persistence_type='session'
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Non-Productive States"),
                    dcc.Dropdown(
                        id='non-productive-states-dropdown',
                        #options=[{'label': state, 'value': state} for state in states],
                        value=[],
                        multi=True,
                        persistence=True,
                        persistence_type='session'
                    ),
                ], width=6),
            ]),
    dbc.Row([
        dbc.Col([
            html.H4([
                "State Cost",
                html.I(className="fas fa-question-circle", id="tooltip-target-cost-state",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-cost-state",
                placement="right",
            )])],style={'margin-top':'30px'}),
    dbc.Row([
                dbc.Col([
                    html.H5("Costs for Productive States"),
                    html.Div(id='productive-cost-inputs',children=[

                    ])
                ], width=6),
                dbc.Col([
                    html.H5("Costs for Non-Productive States"),
                    html.Div(id='non-productive-cost-inputs',children=[

                    ])
                ], width=6),
            ]),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Constraints on Amount of Documents",
                html.I(className="fas fa-question-circle", id="tooltip-target-constraints",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-constraints",
                placement="right",
            )])],style={'margin-top':'30px'}),
    dbc.Row(dbc.Col(html.Div(intervals_table,style={'max-height':'30vh','overflow-y':'auto'}))),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Constraints on Intensities",
                html.I(className="fas fa-question-circle", id="tooltip-target-intensities",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-intensities",
                placement="right",
            )])],style={'margin-top':'30px'}),
    dbc.Row(dbc.Col(html.Div(intensities_table,style={'max-height':'30vh','overflow-y':'auto'}))),
],style={'height':'70vh','overflow':'scroll'})


@callback(
    Output('productive-states-dropdown', 'options'),
    Output('non-productive-states-dropdown', 'options'),
    Output('productive-states','data'),
    Output('unproductive-states','data'),
    Output('states-dict-step2-div','children'),
    Input('productive-states-dropdown', 'value'),
    Input('non-productive-states-dropdown', 'value'),
    Input('data-has-been-loaded','data'),
    State('states','data'),
    State('productive-states-dropdown','options'),
    State('non-productive-states-dropdown','options')
)
def update_state_options(prod_selected, non_prod_selected,data_has_been_loaded,states_stored,prod_options_stored,non_prod_options_stored):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    states = list(states_stored.keys())
    items = [html.Li(f"{key}: {value}") for key, value in states_stored.items()]
    if triggered_id == 'data-has-been-loaded' or not prod_options_stored or not non_prod_options_stored:
        prod_options = [{'label': state, 'value': state} for state in states]
        non_prod_options = [{'label': state, 'value': state} for state in states]
        return prod_options,non_prod_options,[],[],html.Ul(items)
    # Ensure selections are lists
    if prod_selected is None:
        prod_selected = []
    if non_prod_selected is None:
        non_prod_selected = []

    # Validate that no state is selected in both lists
    #common_states = set(prod_selected) & set(non_prod_selected)

    # Update options to exclude selected states from the other dropdown
    available_for_prod = [state for state in states if state not in non_prod_selected]
    available_for_non_prod = [state for state in states if state not in prod_selected]

    prod_options = [{'label': state, 'value': state} for state in available_for_prod]
    non_prod_options = [{'label': state, 'value': state} for state in available_for_non_prod]

    return prod_options, non_prod_options,prod_selected,non_prod_selected,html.Ul(items)

@callback(
    Output('productive-cost-inputs', 'children'),
    Output('non-productive-cost-inputs', 'children'),
    Input('productive-states-dropdown', 'value'),
    Input('non-productive-states-dropdown', 'value'),
    State('state-costs', 'data')
)
def update_cost_inputs(prod_selected, non_prod_selected,state_costs_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    # Generate input fields for productive states

    state_costs_dict = {}
    if state_costs_data is not None:
        state_costs_dict = {item['state']: item['cost'] for item in state_costs_data}

    prod_cost_inputs = []
    for state in prod_selected:
        current_value = state_costs_dict.get(state)
        prod_cost_inputs.append(
            dbc.InputGroup([
                dbc.InputGroupText(f"Cost for {state}"),
                dbc.Input(
                    id={'type': 'prod-cost-input', 'index': state},
                    value=current_value,
                    type='number',
                    min=0,
                    max=100,
                    step=1,
                    placeholder="Enter cost between 0 and 100",
                ),
            ])
        )

    # Generate input fields for non-productive states
    non_prod_cost_inputs = []
    for state in non_prod_selected:
        current_value = state_costs_dict.get(state)
        non_prod_cost_inputs.append(
            dbc.InputGroup([
                dbc.InputGroupText(f"Cost for {state}"),
                dbc.Input(
                    id={'type': 'non-prod-cost-input', 'index': state},
                    value=current_value,
                    type='number',
                    min=0,
                    max=100,
                    step=1,
                    placeholder="Enter cost between 0 and 100",
                ),
            ])
        )

    return prod_cost_inputs, non_prod_cost_inputs

@callback(
    Output('state-costs', 'data'),
    Input({'type': 'prod-cost-input', 'index': ALL}, 'value'),
    Input({'type': 'non-prod-cost-input', 'index': ALL}, 'value'),
    State({'type': 'prod-cost-input', 'index': ALL}, 'id'),
    State({'type': 'non-prod-cost-input', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def collect_state_costs(prod_cost_values, non_prod_cost_values, prod_cost_ids, non_prod_cost_ids):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if triggered_id is not None:
        state_costs = []

        # Collect costs for productive states
        for value, id in zip(prod_cost_values, prod_cost_ids):
            if value is not None:
                state_costs.append({'state': id['index'], 'cost': value})

        # Collect costs for non-productive states
        for value, id in zip(non_prod_cost_values, non_prod_cost_ids):
            if value is not None:
                state_costs.append({'state': id['index'], 'cost': value})

        return state_costs
    else:
        return dash.no_update

@callback(
    Output('intervals-table', 'data'),
    Output('state-constraints', 'data'),
    Input('intervals-table', 'data_timestamp'),
    Input('state-constraints', 'data'),
    Input('data-has-been-loaded','data'),
    State('intervals-table', 'data'),
    State('states','data')
)
def validate_and_correct(timestamp, rows_stored_data, data_has_been_loaded,rows_table_data,states_stored):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    states = list(states_stored.keys())
    if triggered_id == 'data-has-been-loaded':
        rows = get_initial_table_data(states)
    elif triggered_id == 'intervals-table' and rows_table_data:
        rows = rows_table_data
    elif rows_stored_data:
        rows = rows_stored_data
    else:
        rows = get_initial_table_data(states)

    corrected_rows = []
    for row in rows:
        min_val = row.get('min')
        max_val = row.get('max')

        # Validate and correct 'min'
        if min_val is None or min_val < 0 or min_val >= 100:
            row['min'] = 0

        # Validate and correct 'max'
        if max_val is None or max_val <= 0 or max_val > 100:
            row['max'] = 100

        corrected_rows.append(row)

    return corrected_rows, corrected_rows


@callback(
    Output('intensities-table', 'dropdown_conditional'),
    Input('states', 'data'),
)
def update_dropdown(states_stored):
    if not states_stored:
        return dash.no_update

    states = list(states_stored.keys())
    options = [{'label': state, 'value': state} for state in states]

    dropdown_conditional = [
        {
            'if': {
                'column_id': 'from_state'
            },
            'options': options
        },
        {
            'if': {
                'column_id': 'to_state'
            },
            'options': options
        }
    ]
    return dropdown_conditional

@callback(
    Output('intensity-constraints', 'data'),
    Output('intensities-table', 'data'),
    Input('intensities-table', 'data_timestamp'),
    Input('intensity-constraints', 'data'),
    Input('data-has-been-loaded','data'),
    State('intensities-table', 'data'),
    State('states','data'),
    State('intensity-matrix','data')
)
def input_intensity_constraint(timestamp, rows_stored_data, data_has_been_loaded,rows_table_data,states_stored,intensity_matrix_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    states = list(states_stored.keys())
    if triggered_id == 'data-has-been-loaded':
        rows = get_initial_intensity_data(intensity_matrix_data,states)
    elif triggered_id == 'intensities-table' and rows_table_data:
        rows = rows_table_data
    elif rows_stored_data:
        rows = rows_stored_data
    else:
        rows = get_initial_intensity_data(intensity_matrix_data,states)

    return rows,rows