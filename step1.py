import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
from dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go
import utils
import sqlite3
import pandas as pd

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        color = "hsl({}, 100%, 50%)".format(int(360 * i / num_colors))
        colors.append(color)
    return colors


def get_intensity_matrix(dropdown_value=None):
    try:
        conn = sqlite3.connect('application.db')
        df_ = pd.read_sql_query(
            "SELECT * FROM db_table", conn, parse_dates=['state_begin', 'state_end']
        )
        conn.close()

        dropdown_options = []
        dropdown_style = {'display': 'none'}

        if df_.empty:
            return None, None, None, dropdown_options, dropdown_style

        dropdown_options = sorted(df_['document_type'].unique())
        dropdown_style = {'display': 'block'}

        if dropdown_value is not None:
            df_ = df_[df_['document_type'].isin(dropdown_value)]

        if df_.empty:
            return None, None, None, dropdown_options, dropdown_style

        states = sorted(df_['state'].unique())
        state_dict = {f'S{i}': state for i, state in enumerate(states)}
        state_inverse_dict = {state: f'S{i}' for i, state in enumerate(states)}
        N = df_['document_id'].nunique()

        df_ = df_.sort_values(by=['document_id', 'state_begin']).reset_index(drop=True)

        transitions = []
        transition_counts = {}
        non_working_dates = utils.get_non_working_dates()
        total_documents = df_['document_id'].nunique()

        for document_id, group in df_.groupby('document_id'):
            group = group.sort_values(by='state_begin').reset_index(drop=True)
            num_rows = len(group)

            for i in range(num_rows - 1):
                current_row = group.iloc[i]
                next_row = group.iloc[i + 1]
                current_state = current_row['state']
                next_state = next_row['state']

                if current_state != next_state:
                    key = (current_state, next_state)
                    if key not in transition_counts:
                        transition_counts[key] = set()
                    transition_counts[key].add(document_id)

                    if pd.notnull(current_row['state_end']):
                        transition_time = utils.calculate_working_hours(
                            current_row['state_begin'],
                            current_row['state_end'],
                            non_working_dates
                        )
                        transitions.append({
                            'document_id': document_id,
                            'from_state': current_state,
                            'to_state': next_state,
                            'state_end': current_row['state_end'],
                            'state_begin': current_row['state_begin'],
                            'transition_time': transition_time
                        })

        if not transitions:
            return None, None, None, dropdown_options, dropdown_style

        # Calculate the percentage of documents for each transition
        transition_percentages = {
            key: (len(doc_ids) / total_documents) * 100
            for key, doc_ids in transition_counts.items()
        }

        # Filter out transitions that occur in less than 5% of documents
        transitions_to_keep = {
            key for key, percentage in transition_percentages.items() if percentage >= 5
        }

        transitions_df = pd.DataFrame(transitions)

        if transitions_df.empty:
            return None, None, None, dropdown_options, dropdown_style

        # Filter transitions_df to keep only transitions occurring in at least 5% of documents
        transitions_df['transition_key'] = list(zip(
            transitions_df['from_state'], transitions_df['to_state']
        ))
        transitions_df = transitions_df[
            transitions_df['transition_key'].isin(transitions_to_keep)
        ].drop(columns='transition_key')

        if transitions_df.empty:
            return None, None, None, dropdown_options, dropdown_style

        # Filter out transitions with transition_time <= 0.25
        transitions_df = transitions_df[transitions_df['transition_time'] > 0.25]

        if transitions_df.empty:
            return None, None, None, dropdown_options, dropdown_style

        # Calculate mean transition times and intensities
        mean_transition_times = transitions_df.groupby(
            ['from_state', 'to_state']
        )['transition_time'].mean().reset_index()
        mean_transition_times['intensity'] = 1 / mean_transition_times['transition_time']

        # Initialize intensity matrix with zeros
        intensity_matrix = pd.DataFrame(
            index=states, columns=states, data=0.0
        )

        # Populate intensity matrix
        for _, row in mean_transition_times.iterrows():
            from_state = row['from_state']
            to_state = row['to_state']
            intensity = row['intensity']
            intensity_matrix.loc[from_state, to_state] = intensity

        # Rename columns and indices using state_inverse_dict
        intensity_matrix.rename(columns=state_inverse_dict, inplace=True)
        intensity_matrix.index = intensity_matrix.index.map(state_inverse_dict)

        # Prepare the intensity matrix for output
        matrix = intensity_matrix.reset_index().rename(columns={'index': 'State'})

        return matrix, state_dict, N, dropdown_options, dropdown_style

    except Exception as e:
        # Log the exception if needed
        # print(f"Error in get_intensity_matrix: {e}")
        return None, None, None, [], {'display': 'none'}


def get_intensity_matrix_old(dropdown_value=None):
    try:
        conn = sqlite3.connect('application.db')
        df_ = pd.read_sql_query("SELECT * FROM db_table", conn, parse_dates=['state_begin', 'state_end'])
        dropdown_options = []
        dropdown_style = {'display':'none'}
        if df_.empty:
            return None, None, None, dropdown_options, dropdown_style
        dropdown_options = sorted(df_['document_type'].unique())
        dropdown_style = {'display':'block'}
        if dropdown_value is not None:
            df_ = df_[df_['document_type'].isin(dropdown_value)]
        states = sorted(df_['state'].unique())
        state_dict = {f'S{i}': state for i, state in enumerate(states)}
        state_inverse_dict = {state: f'S{i}' for i, state in enumerate(states)}
        N = len(df_['document_id'].unique())
        df_ = df_.sort_values(by=['document_id', 'state_begin']).reset_index(drop=True)
        transitions = []
        non_working_dates = utils.get_non_working_dates()
        for document_id, group in df_.groupby('document_id'):
            group = group.sort_values(by='state_begin').reset_index(drop=True)
            num_rows = len(group)

            for i in range(num_rows):
                current_row = group.iloc[i]
                current_state = current_row['state']

                for j in range(i + 1, num_rows):
                    next_row = group.iloc[j]
                    next_state = next_row['state']

                    if current_state != next_state:
                        count = 1
                        k = j + 1
                        while k < num_rows and group.iloc[k]['state'] == next_state:
                            count += 1
                            k += 1

                        if pd.notnull(current_row['state_end']):
                            for _ in range(count):
                                transition_time = utils.calculate_working_hours(
                                    current_row['state_begin'],
                                    current_row['state_end'],
                                    non_working_dates
                                )

                                transitions.append({
                                    'document_id': document_id,
                                    'from_state': current_state,
                                    'to_state': next_state,
                                    'state_end': current_row['state_end'],
                                    'state_begin': current_row['state_begin'],
                                    'transition_time': transition_time
                                })
                        break

        if not transitions:
            return None, None, None, dropdown_options, dropdown_style

        transitions_df = pd.DataFrame(transitions)

        transitions_df = transitions_df[transitions_df['transition_time'] > 0.25]
        mean_transition_times = transitions_df.groupby(['from_state', 'to_state'])['transition_time'].mean().reset_index()
        mean_transition_times['intensity'] = 1 / mean_transition_times['transition_time']
        intensity_matrix = pd.DataFrame(index=states, columns=states, data=0.0)

        for _, row in mean_transition_times.iterrows():
            from_state = row['from_state']
            to_state = row['to_state']
            intensity = row['intensity']
            intensity_matrix.loc[from_state, to_state] = intensity

        intensity_matrix.rename(columns=state_inverse_dict, inplace=True)
        intensity_matrix.index = intensity_matrix.columns
        matrix = intensity_matrix.reset_index().rename(columns={'index': 'State'})
        return matrix, state_dict, N, dropdown_options, dropdown_style
    except Exception as e:
        return None,None,None,[],{'display':'none'}

def get_default_intensity_matrix():
    data = {
        'Fill in the document': [0, 0.03, 0.07, 0.06, 0],
        'Approval': [0.15, 0, 0, 0, 0],
        'Signing': [0, 0.04, 0, 0, 0],
        'Execution': [0, 0, 0.13, 0, 0.01],
        'Archiving': [0, 0, 0, 0.005, 0],
    }

    df = pd.DataFrame(data)

    state_dict = {'S0': 'Fill in the document',
                  'S1': 'Approval',
                  'S2': 'Signing',
                  'S3': 'Execution',
                  'S4': 'Archiving'}
    state_inverse_dict = {value: key for key, value in state_dict.items()}

    df.rename(columns=state_inverse_dict, inplace=True)
    df.index = df.columns
    matrix = df.reset_index().rename(columns={'index': 'State'})
    N = 1500
    return matrix,state_dict,N

def load_intensity_matrix(dropdown_value=None):
    matrix,state_dict,N,dropdown_options,dropdown_style = get_intensity_matrix(dropdown_value)
    if matrix is None:
        matrix,state_dict,N = get_default_intensity_matrix()
        dropdown_options = []
        dropdown_style = {'display':'none'}
    return matrix,state_dict,N,dropdown_options,dropdown_style

def check_strong_connectivity(intensity_matrix):
    adjacency_matrix = (intensity_matrix > 0).astype(int)

    states = intensity_matrix.index.tolist()
    num_states = len(states)

    adjacency_list = {state: [] for state in states}
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if adjacency_matrix.iloc[i, j]:
                adjacency_list[from_state].append(to_state)

    def dfs(current_state, visited, adjacency_list):
        visited.add(current_state)
        for neighbor in adjacency_list[current_state]:
            if neighbor not in visited:
                dfs(neighbor, visited, adjacency_list)

    for start_state in states:
        visited = set()
        dfs(start_state, visited, adjacency_list)
        if len(visited) != num_states:
            return False

    return True


def compute_state_dynamics(intensity_matrix, states, initial_state, N, time_span):
    if not initial_state:
        initial_state = states[0]
    m0 = np.zeros(len(states))
    m0[list(states).index(initial_state)] = N

    t_span = (0, time_span)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    Q = intensity_matrix.values

    m0 = m0.astype(float)

    def average_dynamics(t, m):
        return np.dot(Q.T, m)

    solution = solve_ivp(
        fun=average_dynamics,
        t_span=t_span,
        y0=m0,
        t_eval=t_eval,
        method='RK45'
    )

    fig = go.Figure()

    colors = generate_colors(len(states))
    for idx, state in enumerate(states):
        fig.add_trace(go.Scatter(
            x=solution.t,
            y=solution.y[idx],
            mode='lines',
            name=f'State {state}',
            line=dict(color=colors[idx])
        ))

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Expected Number of Documents',
        title='',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=0),
        legend=dict(
            title='States',
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified',
        autosize=True,
    )
    fig.update_xaxes(
        automargin=False,
        constrain='domain',
        ticklabelposition="inside top"
    )
    fig.update_yaxes(
        automargin=False,
        constrain='domain'
    )

    return fig


def calculate_stationary_distribution(intensity_matrix, states, N):
    is_strongly_connected = check_strong_connectivity(intensity_matrix)

    if is_strongly_connected:
        Q = intensity_matrix.values
        QT = Q.T
        num_states = len(states)
        A_eq = np.vstack([QT, np.ones(num_states)])
        b_eq = np.zeros(num_states + 1)
        b_eq[-1] = N
        bounds = [(0, None) for _ in range(num_states)]
        res = linprog(
            c=np.zeros(num_states),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        if res.success:
            m_initial = res.x
        else:
            m_initial = np.zeros(num_states)
        m_initial_data = [{
            'state': state,
            'm_initial': round(m_val)
        } for state, m_val in zip(states, m_initial)]
    else:
        m_initial_data = []

    return m_initial_data, is_strongly_connected

layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div(
                [html.H4("Document Types"),
                dcc.Dropdown(
                        id='step1-document-type-dropdown',
                        options=[],
                        multi=True
                    ),
            ],id='step1-document-type-dropdown-div',style={'display':'none'})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4([
                "States Dictionary",
                html.I(className="fas fa-question-circle", id="tooltip-target-states-dict-step1", style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
    "This tooltip provides additional information about the intensity matrix.",
    target="tooltip-target-states-dict-step1",
    placement="right",
            )])],style={'margin-top':'15px'}),
    dbc.Row([
        dbc.Col([
            html.Div(id='states-dict-step1-div',style={'max-height':'15vh','overflow-y':'auto','margin-bottom':'15px'}),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Intensity Transition Matrix",
                html.I(className="fas fa-question-circle", id="tooltip-target-intensity-matrix", style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
    "This tooltip provides additional information about the intensity matrix.",
    target="tooltip-target-intensity-matrix",
    placement="right",
            )])]),
    dbc.Row(dbc.Col(html.Div(
        dash_table.DataTable(
                id='intensity-matrix-table',
        style_data_conditional=[
        {
            'if': {'column_id': 'State'},
            'backgroundColor': '#ddd',
            'color': 'black',
            'font-weight': 'bold',
            'text-align': 'center',
        }],
            ),
    ))),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Dynamics of the States",
                html.I(className="fas fa-question-circle", id="tooltip-target-dynamics",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-dynamics",
                placement="right",
            )])],style={'margin-top':'30px'}),
    dbc.Row([
        dbc.Col([
            html.Label("Initial State:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='initial-state-dropdown',
                clearable=False,
                style={'minWidth': '200px'}
            ),
        ], width='auto', style={'display': 'flex', 'alignItems': 'center', 'marginRight': '20px'}),
        dbc.Col([
            html.Label("Modelling Time:", style={'marginRight': '10px'}),
            dbc.Input(
                id='time-span-input',
                type='number',
                value=100,
                min=0,
                step=5,
                style={'width': '100px'}
            ),
        ], width='auto', style={'display': 'flex', 'alignItems': 'center'}),
    ], justify='start', align='center'),
    dbc.Row(dbc.Col(html.Div(children=[
            html.H5('Expected Counts of Documents in Each State')]),width='auto'),justify='center'),
    dbc.Row(
        dbc.Col(
            html.Div(children=[
            dcc.Graph(id='state-dynamics-graph'),
            ], style={'padding': '0', 'margin': '0', 'height': '100%', 'width': '100%'})
        ),
    ),
    dbc.Row([
        dbc.Col([
            html.H4([
                "Stationary Distribution",
                html.I(className="fas fa-question-circle", id="tooltip-target-stationary", style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
    "This tooltip provides additional information about the intensity matrix.",
    target="tooltip-target-stationary",
    placement="right",
            )])]),
    dbc.Row(dbc.Col(html.Div(id='check-stationary-distr',style={'margin-bottom':'15px'}))),
    dbc.Row(dbc.Col(html.Div(
        dash_table.DataTable(
                id='m-initial-table',
                columns=[
                    {'name': 'State', 'id': 'state'},
                    {'name': 'Document Count', 'id': 'm_initial', 'type': 'numeric', 'format': {'specifier': '.0f'}}
                ],
                data=[],
            ),style={'max-height':'30vh','overflow-y':'auto','margin-bottom':'20px'}
    )))
],style={'height':'70vh','overflow':'scroll'})

@callback(
    Output('state-dynamics-graph', 'figure'),
    Output('m-initial-table', 'data'),
    Output('intensity-matrix','data'),
    Output('stationary-distribution','data'),
    Output('intensity-matrix-table', 'data'),
    Output('initial-state-dropdown','options'),
    Output('states', 'data'),
    Output('N', 'data'),
    Output('intensity-matrix-table', 'columns'),
    Output('states-dict-step1-div','children'),
    Output('check-stationary-distr','children'),
    Output('step1-document-type-dropdown', 'options'),
    Output('step1-document-type-dropdown-div', 'style'),
    Output('doc-type-val', 'data'),
    Output('step1-document-type-dropdown', 'value'),
    Input('intensity-matrix-table', 'data'),
    Input('initial-state-dropdown', 'value'),
    Input('time-span-input', 'value'),
    Input('data-has-been-loaded','data'),
    Input('step1-document-type-dropdown', 'value'),
    State('doc-type-val', 'data'),
)
def update_state_dynamics(intensity_table_data, initial_state, time_span,data_has_been_loaded,dropdown_value_trigger,doc_type_stored):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if dropdown_value_trigger is not None:
        doc_type = dropdown_value_trigger
    elif doc_type_stored:
        doc_type = doc_type_stored
    else:
        doc_type = None

    intensity_matrix, states_dict, N,dropdown_options,dropdown_style = load_intensity_matrix(doc_type)
    states = list(states_dict.keys())
    if triggered_id == 'intensity-matrix-table' and intensity_table_data:
        intensity_data = intensity_table_data
    else:
        intensity_data = intensity_matrix.to_dict('records')

    intensity_matrix_df = pd.DataFrame(intensity_data)
    columns = []
    for col in intensity_matrix_df.columns:
        if col == 'State':
            columns.append({'name': col, 'id': col})
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'editable':True,
                'format': Format(precision=2, scheme=Scheme.fixed)
            })
    intensity_matrix_df.set_index('State', inplace=True)
    intensity_matrix = intensity_matrix_df.astype(float)

    if not initial_state:
        initial_state = states[0]
    for state in states:
        intensity_matrix.loc[state, state] = -intensity_matrix.loc[state].sum()

    fig = compute_state_dynamics(intensity_matrix.copy(), states, initial_state, N, time_span)

    m_initial_data, is_strongly_connected = calculate_stationary_distribution(intensity_matrix.copy(), states, N)

    div_check_stationary_distrib = []
    if not is_strongly_connected:
        div_check_stationary_distrib.append('Strong connectivity of states is not observed which is necessary for the existence of a unique stationary distribution')

    # Update intensity-matrix-table data
    options = [{'label': state, 'value': state} for state in states]
    items = [html.Li(f"{key}: {value}") for key, value in states_dict.items()]


    return fig, m_initial_data, intensity_data, m_initial_data, intensity_data, options,states_dict,N,columns,html.Ul(items),div_check_stationary_distrib,dropdown_options,dropdown_style,doc_type,doc_type
