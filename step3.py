import dash
from dash import html, dcc, Input, Output, State, callback, ctx, dash_table, ALL
import dash_bootstrap_components as dbc
import pandas as pd

def create_strategy_card(strategy_number, selected_states, states, is_saved=False):
    card_header = dbc.CardHeader([
        html.H5(f"Strategy {strategy_number}", style={'display': 'inline-block', 'margin-right': '10px'}),
        dbc.Button(
            "Delete",
            id={'type': 'delete-strategy-button', 'index': strategy_number},
            color="danger",
            size="sm",
            n_clicks=0,
            style={'float': 'right'}
        ),
    ])
    card_body_children = [
        dcc.Dropdown(
            id={'type': 'strategy-states-dropdown', 'index': strategy_number},
            options=[{'label': state, 'value': state} for state in states],
            value=selected_states,
            multi=True,
            placeholder="Select states for this strategy",
        ),
    ]
    if not is_saved:
        card_body_children.append(
            dbc.Button(
                "Save",
                id={'type': 'save-strategy-button', 'index': strategy_number},
                color="primary",
                size="sm",
                n_clicks=0,
                style={'margin-top': '10px'}
            )
        )

    card_body = dbc.CardBody(card_body_children)

    return dbc.Card([
        card_header,
        card_body,
    ], style={'margin-bottom': '15px'}, id={'type': 'strategy-card', 'index': strategy_number})

layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4([
                "States Dictionary",
                html.I(className="fas fa-question-circle", id="tooltip-target-states-dict-step3", style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
    "This tooltip provides additional information about the intensity matrix.",
    target="tooltip-target-states-dict-step3",
    placement="right",
            )])]),
    dbc.Row([
        dbc.Col([
            html.Div(id='states-dict-step3-div',style={'max-height':'15vh','overflow-y':'auto','margin-bottom':'15px'}),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Strategies",
                html.I(className="fas fa-question-circle", id="tooltip-target-strategies",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-strategies",
                placement="right",
            )])]),
    dbc.Row(dbc.Col(html.Div(
            [dbc.Button("Add Strategy", id="add-strategy-button", color="primary", n_clicks=0),
            html.Br(), html.Br(),
            html.Div(id="strategies-container")]
    )))


],style={'height':'70vh','overflow':'scroll'})


@callback(
    Output('strategies-container', 'children'),
    Output('states-dict-step3-div','children'),
    Input('strategies-store', 'data'),
    Input('data-has-been-loaded','data'),
    State('states','data')
)
def render_strategy_cards(strategies,data_has_been_loaded,states_stored):
    cards = []
    states = list(states_stored.keys())
    items = [html.Li(f"{key}: {value}") for key, value in states_stored.items()]

    if strategies:
        for strategy in strategies:
            card = create_strategy_card(
                strategy['number'],
                strategy['states'],
                states,
                is_saved=strategy['is_saved']
            )
            cards.append(card)
        return cards,html.Ul(items)
    else:
        return dash.no_update,html.Ul(items)



# Callback to add a new strategy card
@callback(
    Output('strategies-store', 'data',allow_duplicate=True),
    Input('add-strategy-button', 'n_clicks'),
    State('strategies-container', 'children'),
    State('strategies-store', 'data'),
    prevent_initial_call=True
)
def add_strategy(n_clicks, existing_cards, strategies):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    if n_clicks > 0:
        # Assign a new strategy number
        strategy_number = len(strategies) + 1

        # Add the new strategy to the store
        strategies.append({
            'number': strategy_number,
            'states': [],
            'is_saved': False,
        })

        return strategies
    else:
        return dash.no_update

@callback(
    Output('strategies-store', 'data', allow_duplicate=True),
    Input({'type': 'save-strategy-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'strategy-states-dropdown', 'index': ALL}, 'value'),
    State('strategies-store', 'data'),
    prevent_initial_call=True
)
def save_strategy(n_clicks_list, states_list, strategies):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    # Identify which save button was clicked
    triggered_prop = ctx.triggered[0]['prop_id']
    button_id = eval(triggered_prop.split('.')[0])
    triggered_index = button_id['index']

    # Find the index of the clicked button in n_clicks_list
    button_indices = [btn['id']['index'] for btn in ctx.inputs_list[0]]
    idx_clicked = button_indices.index(triggered_index)

    # Check if the n_clicks of the triggered button is > 0
    if n_clicks_list[idx_clicked] > 0:
        for i, strategy in enumerate(strategies):
            if strategy['number'] == triggered_index:
                selected_states = states_list[i]
                # Validation: Ensure the set of states is unique among saved strategies
                existing_states_sets = [set(s['states']) for s in strategies if s['is_saved']]
                if set(selected_states) in existing_states_sets:
                    # Validation failed: Duplicate set of states
                    return strategies
                else:
                    # Save the strategy
                    strategies[i]['states'] = selected_states
                    strategies[i]['is_saved'] = True
                    break
        return strategies
    else:
        return dash.no_update

@callback(
    Output('strategies-store', 'data',allow_duplicate=True),
    Input({'type': 'delete-strategy-button', 'index': ALL}, 'n_clicks'),
    State('strategies-store', 'data'),
    prevent_initial_call=True
)
def delete_strategy(n_clicks_list, strategies):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    # Identify which delete button was clicked
    triggered_prop = ctx.triggered[0]['prop_id']
    button_id = eval(triggered_prop.split('.')[0])
    triggered_index = button_id['index']

    # Find the index of the clicked button in n_clicks_list
    button_indices = [btn['id']['index'] for btn in ctx.inputs_list[0]]
    idx_clicked = button_indices.index(triggered_index)

    # Check if the n_clicks of the triggered button is > 0
    if n_clicks_list[idx_clicked] > 0:
        # Remove the strategy from the list
        strategies = [s for s in strategies if s['number'] != triggered_index]

        # Reassign strategy numbers sequentially starting from 1
        for idx, strategy in enumerate(strategies):
            strategy['number'] = idx + 1

        return strategies
    else:
        return dash.no_update






