import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import utils

import pandas as pd

dash.register_page(__name__, path='/monitoring')

df = pd.read_excel('test.xlsx')
replacements = {
    '04.1 Заявка на командировку': '04.1 Business Trip Request',
    '02.01 Акт / Товарная накладная / УПД': '02.01 Act / Goods Invoice / Universal Transfer Document',
    '03.1 Общая служебная записка': '03.1 General Internal Memo',
    '02.03 Счет': '02.03 Invoice',
    '01.1 Договор ': '01.1 Contract',
    '03.2 Финансовая (экономическая) служебная записка': '03.2 Financial Internal Memo',
    '01.2 Дополнение/ изменение к Договору ': '01.2 Contract Amendment/Modification'
}
#df['document_type'] = df['document_type'].replace(replacements)
df['state_begin'] = pd.to_datetime(df['state_begin'],errors='coerce')
df['state_end'] = pd.to_datetime(df['state_end'],errors='coerce')
df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
filtered_df = df[
    (df['state_type'] == 'Approve') &
    (df['state'].str.contains('Approval')) &
    (df['state_finished'] == 'Да')
].copy()
non_working_dates = utils.get_non_working_dates()
df['delay'] = df.apply(
    lambda row: utils.calculate_working_hours(row['deadline'], row['state_end'], non_working_dates), axis=1)

document_types = filtered_df['document_type'].unique()
approval_stages = filtered_df['state'].unique()
metric_mapping = {
    'Workload': 'executor_workload',
    'Percentage of non-approvals': 'executor_percentage_of_non_approvals',
    'Percentage of approvals': 'executor_percentage_of_approvals',
    'Percentage of delays': 'executor_percentage_of_delays',
    'Mean delay': 'executor_mean_delay'
}
colors_palette = ['#ADD8E6','#B0C4DE','#AFDAFC','#A2A2D0']

def amount_of_documents(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)
    return df_.loc[mask, 'document_id'].nunique()


def percentage_non_approvals(df_, document_type_filter=None, state_filter=None):
    mask = df_['state_iteration'] == 1
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)
    temp_df = df_.loc[mask]

    grouped = temp_df.groupby(['document_id', 'state'])['approval_result'].first()

    total = len(grouped)
    if total == 0:
        return 0

    non_approved = (grouped == 'Не согласовано').sum()
    percentage = (non_approved / total) * 100
    return percentage


def percentage_of_delays(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)

    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    filtered_delays = df_.loc[mask, 'delay']

    if filtered_delays.empty:
        return 0

    delayed_count = (filtered_delays > 0).sum()
    total_count = len(filtered_delays)

    percentage = (delayed_count / total_count) * 100

    return percentage

def average_and_max_delay(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)

    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)

    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    delays = df_.loc[mask, 'delay']

    positive_delays = delays[delays > 0]

    if positive_delays.empty:
        return 0, 0

    avg_delay = positive_delays.mean()
    max_delay = positive_delays.max()

    return avg_delay, max_delay


def average_and_max_iterations(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)

    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    temp_df = df_.loc[mask]

    if temp_df.empty:
        return 0, 0

    iterations = temp_df.groupby(['document_id', 'state'])['state_iteration'].max()

    avg_iterations = iterations.mean()
    max_iterations = iterations.max()

    return avg_iterations, max_iterations


def get_state_mean_delay(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)

    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)

    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    mask &= df_['delay'] > 0

    delays = df_.loc[mask, ['state', 'delay']]

    if delays.empty:
        return pd.DataFrame(columns=['state', 'delay'])

    state_mean_delay = delays.groupby('state')['delay'].mean().reset_index()

    return state_mean_delay


def states_percentage_non_approvals(df_, document_type_filter=None, state_filter=None):
    mask = df_['state_iteration'] == 1

    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    temp_df = df_.loc[mask, ['state', 'document_id', 'approval_result']]

    if temp_df.empty:
        return pd.DataFrame(columns=['state', 'non_approval_percentage'])

    temp_df = temp_df.drop_duplicates(subset=['state', 'document_id'], keep='first')

    approval_pivot = temp_df.pivot_table(
        index='state',
        columns='approval_result',
        values='document_id',
        aggfunc='count',
        fill_value=0
    )

    approval_pivot['total_documents'] = approval_pivot.sum(axis=1)

    approval_pivot['non_approval_percentage'] = (
        approval_pivot.get('Не согласовано', 0) / approval_pivot['total_documents'] * 100
    )

    result_df = approval_pivot.reset_index()[['state', 'non_approval_percentage']]

    result_df = result_df[result_df['non_approval_percentage'] != 0]

    return result_df


def delay_categories(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)

    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    delay_series = df_.loc[mask, 'delay']

    if delay_series.empty:
        return {
            '<8 hours': 0,
            '8-16 hours': 0,
            '16-24 hours': 0,
            '>24 hours': 0
        }

    categories = {
        '<8 hours': ((delay_series > 0) & (delay_series <= 8)).sum(),
        '8-16 hours': ((delay_series > 8) & (delay_series <= 16)).sum(),
        '16-24 hours': ((delay_series > 16) & (delay_series <= 24)).sum(),
        '>24 hours': (delay_series > 24).sum()
    }
    return categories

def max_iterations_counts(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    filtered_df = df_.loc[mask]

    if filtered_df.empty:
        return pd.DataFrame(columns=['max_iteration', 'document_count'])

    max_iterations = filtered_df.groupby('document_id')['state_iteration'].max()

    max_iteration_counts = max_iterations.value_counts().reset_index()
    max_iteration_counts.columns = ['max_iteration', 'document_count']

    max_iteration_counts = max_iteration_counts.sort_values('max_iteration')

    return max_iteration_counts

def states_max_iterations(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    filtered_df = df_.loc[mask]

    if filtered_df.empty:
        return pd.DataFrame(columns=['state', 'max_state_iteration'])

    max_iterations = filtered_df.groupby(['state', 'document_id'])['state_iteration'].max()

    state_max_iter = max_iterations.groupby('state').max().reset_index()

    state_max_iter.rename(columns={'state_iteration': 'max_state_iteration'}, inplace=True)

    return state_max_iter

def states_mean_max_iterations(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    filtered_df = df_.loc[mask]

    if filtered_df.empty:
        return pd.DataFrame(columns=['state', 'mean_max_iteration'])

    max_iterations = filtered_df.groupby(['state', 'document_id'])['state_iteration'].max()

    state_mean_iter = max_iterations.groupby('state').mean().reset_index()

    state_mean_iter.rename(columns={'state_iteration': 'mean_max_iteration'}, inplace=True)

    state_mean_iter = state_mean_iter[state_mean_iter['mean_max_iteration'] > 1]

    return state_mean_iter


def calculate_loop_percentage(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    filtered_df = df_.loc[mask, ['document_id', 'state_iteration']]

    total_documents = filtered_df['document_id'].nunique()

    if total_documents == 0:
        return 0, 0, 0

    loop_documents = filtered_df.loc[filtered_df['state_iteration'] > 1, 'document_id'].nunique()
    count_loop_documents = loop_documents

    percentage = (count_loop_documents / total_documents) * 100

    return count_loop_documents, total_documents, percentage

def executor_metrics(df, document_type_filter=None, state_filter=None):
    temp_df = df.copy()
    if document_type_filter:
        temp_df = temp_df[temp_df['document_type'].isin(document_type_filter)]
    if state_filter:
        temp_df = temp_df[temp_df['state'].isin(state_filter)]


    executor_group = temp_df.groupby(['executor','executor_department']).agg(
        executor_workload=('document_id', 'count'),
        executor_percentage_of_delays=('delay', lambda x: (x > 0).mean() * 100),
        executor_percentage_of_non_approvals=('approval_result', lambda x: (x == 'Не согласовано').mean() * 100),
        executor_percentage_of_approvals=(
        'approval_result', lambda x: (x.isin(['Согласовано', 'Согласовано с замечаниями'])).mean() * 100),
        executor_mean_delay=('delay', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()

    return executor_group

def approval_result_distribution(df_, document_type_filter=None, state_filter=None):
    mask = pd.Series(True, index=df_.index)
    if document_type_filter is not None:
        mask &= df_['document_type'].isin(document_type_filter)
    if state_filter is not None:
        mask &= df_['state'].isin(state_filter)

    filtered_results = df_.loc[mask, 'approval_result']

    if filtered_results.empty:
        return pd.DataFrame(columns=['approval_result', 'count'])

    approval_result_translated = filtered_results.replace({
        'Согласовано': 'Approved',
        'Не согласовано': 'Not Approved',
        'Согласовано с замечаниями': 'Approved with comments'
    })

    distribution = approval_result_translated.value_counts().reset_index()
    distribution.columns = ['approval_result', 'count']

    return distribution

def generate_slider_for_scatter_plot():
    df_executors = executor_metrics(df)
    return html.Div(
        id="slider-control",
        style={'marginTop': 20},
        children=[
            html.P("Workload constraint"),
            dcc.RangeSlider(
                id='range-slider',
                min=0,
                max=df_executors['executor_workload'].max(),
                step=25,
                value=[0, df_executors['executor_workload'].max()],
                allowCross=False
            ),
        ]
    )

layout = html.Div([
dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        html.H6('% of non-approvals',style={'text-align':'center'})
                    ),
                    html.Div(
                        dcc.Graph(
                            id='non-approvals-gauge',
                            config={'displayModeBar': False},
                            style={'height': '100%', 'width': '100%'}
                        ),
                        className='p-0 d-flex align-items-center h-100'
                    ),
                ],
                className='chart-card h-100'
            ),
            width=2,
            className='h-100'
        ),

        dbc.Col(
            html.Div(
                [
                    html.Div(
                        html.H6('% of delays',style={'text-align':'center'})
                    ),
                    html.Div(
                        dcc.Graph(
                            id='percentage-delays-gauge',
                            config={'displayModeBar': False},
                            style={'height': '100%', 'width': '100%'}
                        ),
                        className='p-0 d-flex align-items-center h-100'
                    ),
                ],
                className='chart-card h-100'
            ),
            width=2,
            className='h-100'
        ),

        dbc.Col(
            html.Div(
                [
                    html.Div(
                        html.H6('average delay',style={'text-align':'center'})
                    ),
                    html.Div(
                        dcc.Graph(
                            id='average-delay-gauge',
                            config={'displayModeBar': False},
                            style={'height': '100%', 'width': '100%'}
                        ),
                        className='p-0 d-flex align-items-center h-100'
                    ),
                ],
                className='chart-card h-100'
            ),
            width=2,
            className='h-100'
        ),

        dbc.Col(
            html.Div(
                [
                    html.Div(
                        html.H6('mean iteration number',style={'text-align':'center'})
                    ),
                    html.Div(
                        dcc.Graph(
                            id='average-iterations-gauge',
                            config={'displayModeBar': False},
                            style={'height': '100%', 'width': '100%'}
                        ),
                        className='p-0 d-flex align-items-center h-100'
                    ),
                ],
                className='chart-card h-100'
            ),
            width=2,
            className='h-100'
        ),

        dbc.Col(
            html.Div(
                [
                    html.Div(
                        html.H6('reverse document movement',style={'text-align':'center'})
                    ),
                    html.Div(
                        dcc.Graph(
                            id='reverse-motion-gauge',
                            config={'displayModeBar': False},
                            style={'height': '100%', 'width': '100%'}
                        ),
                        className='p-0 d-flex align-items-center h-100'
                    ),
                ],
                className='chart-card h-100'
            ),
            width=2,
            className='h-100'
        ),
    ],
    justify='between',
    className="mb-4",
    style={'height': '15vh'}
),
    dbc.Row([
        dbc.Col([

            html.H5('Date Range', style={'margin': '10px'}),
            html.Div([dcc.DatePickerRange(
                id='date-range-picker',
                min_date_allowed=df['state_begin'].min(),
                max_date_allowed=df['state_begin'].max(),
                initial_visible_month=df['state_begin'].min(),
                start_date=df['state_begin'].min(),
                end_date=df['state_begin'].max(),
                display_format='DD-MM-YYYY'
            )]),
            html.H5('Document Type', style={'margin': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='document-type-dropdown',
                    options=[{'label': doc_type, 'value': doc_type} for doc_type in document_types],
                    multi=True,
                    value=document_types,
                    className='complicated-dropdown',
                    optionHeight=55
                )
            ], style={
                'height': '25vh',
                'overflow-y': 'auto'
            }),

            html.Br(),

            html.H5('Approval Stage', style={'margin': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='approval-stage-dropdown',
                    options=[{'label': stage, 'value': stage} for stage in approval_stages],
                    multi=True,
                    value=approval_stages,
                    optionHeight=55
                )
            ], style={
                'height': '25vh',
                'overflow-y': 'auto'
            }),
        ], width=2, style={
            'padding': '10px'
        }),


        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('Average delays by approval stages',style={'text-align':'center'}),
                        dcc.Loading(dcc.Graph(id='average-delay-bar',style={"height": "300px"}))
                    ],className="chart-card"),
                ], width=7),
                dbc.Col([
                    html.Div([
                        html.H6('Delay distribution',style={'text-align':'center'}),
                        dcc.Loading(dcc.Graph(id='delay-categories-pie',style={"height": "300px"}))
                    ],className="chart-card"),

                ], width=5),
    ],style={'margin':'20px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('Percentage of non-approvals by approval stages',style={'text-align':'center'}),
                        dcc.Loading(dcc.Graph(id='non-approval-states-bar',style={"height": "300px"}))
                    ],className="chart-card"),
                ], width=7),
                dbc.Col([
                    html.Div([
                        html.H6('Loop distribution',style={'text-align':'center'}),
                        dcc.Loading(dcc.Graph(id='loop-states-bar',style={"height": "300px"}))
                    ],className="chart-card"),

                ], width=5),
    ],style={'margin':'20px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('Mean iteration number by approval stages',style={'text-align':'center'}),
                        dcc.Loading(dcc.Graph(id='mean-loop-states-bar',style={"height": "300px"}))
                    ],className="chart-card"),

                ], width=7),
                dbc.Col([
                    html.Div([
                        html.H6('Approval result distribution', style={'text-align': 'center'}),
                        dcc.Loading(dcc.Graph(id='approval-result-pie-chart',style={"height": "300px"}))
                    ],className="chart-card"),

                ], width=5),
    ],style={'margin':'20px'}),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        html.Div([
                            html.Label("X-axis: "),
                            dcc.Dropdown(
                                id="metric-select-x",
                                multi=False,
                                optionHeight=55,
                                searchable=True,
                                options=[{"label": i, "value": i} for i in list(metric_mapping.keys())],
                                value=list(metric_mapping.keys())[0],
                                style={'width': '100%'}
                            ),
                        ])
                    ]),
                    dbc.Row([
                        html.Div([
                            html.Label("Y-axis: "),
                            dcc.Dropdown(
                                id="metric-select-y",
                                multi=False,
                                optionHeight=55,
                                searchable=True,
                                options=[{"label": i, "value": i} for i in list(metric_mapping.keys())],
                                value=list(metric_mapping.keys())[1],
                                style={'width': '100%'}
                            ),
                        ])
                    ]),
                    dbc.Row([
                        generate_slider_for_scatter_plot()
                    ]),
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H6('Executive discipline',style={'text-align':'center'}),
                        dcc.Loading(dcc.Graph(id='workers-stats', style={"height": "300px"}))
                    ],className="chart-card", ),

                ], width=10),
            ], style={'margin': '20px'}),
        ], width=10, style={'height': '70vh','overflow-y':'scroll'})
    ],style={'margin-top':'15px'})
],style={'margin':'15px 30px'})

@callback(
    Output("workers-stats", "figure"),
    [
        Input("range-slider", "value"),
        Input("metric-select-x", "value"),
        Input("metric-select-y", "value")
    ],
)
def update_bar_chart(slider_range, metric_x, metric_y):
    df_executors = executor_metrics(df)

    low, high = slider_range
    mask = (df_executors['executor_workload'] >= low) & (df_executors['executor_workload'] <= high)
    df_filtered = df_executors[mask]

    x_col = metric_mapping[metric_x]
    y_col = metric_mapping[metric_y]

    blue_palette = px.colors.sequential.Blues

    fig = px.scatter(
        df_filtered,
        x=x_col,
        y=y_col,
        color="executor_department",
        size='executor_workload',
        hover_name='executor',
        hover_data=[y_col],
        color_discrete_sequence=blue_palette
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            gridcolor='#dcdcdc',
            zerolinecolor='#dcdcdc'
        ),
        yaxis=dict(
            gridcolor='#dcdcdc',
            zerolinecolor='#dcdcdc'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig


@callback(
    Output('average-delay-bar', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_average_delay_bar(document_types, approval_stages):
    state_mean_delay_df = get_state_mean_delay(
        df,
        document_type_filter=document_types,
        state_filter=approval_stages
    )

    if not state_mean_delay_df.empty:
        state_mean_delay_df = state_mean_delay_df.sort_values(by='delay', ascending=False)

        fig = go.Figure(go.Bar(
            x=state_mean_delay_df['delay'],
            y=state_mean_delay_df['state'],
            orientation='h',
            marker=dict(
                color=colors_palette[0],
                line=dict(
                    color='#dcdcdc',
                    width=1
                )
            ),
        ))
        fig.update_layout(
            xaxis_title='Average Delay (hours)',
            yaxis_title='Approval Stage',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True,
            height=None,
            width=None,
            yaxis=dict(
                autorange="reversed",
                gridcolor='#dcdcdc'
            ),
            xaxis=dict(
                gridcolor='#dcdcdc'
            )
        )
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(
            title='Average Delay by Approval Stage (in hours)',
            xaxis_title='Average Delay (hours)',
            yaxis_title='Approval Stage',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),

        )
        fig.add_annotation(
            dict(
                text="No data available for the selected filters",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=20,
                    color="red"
                )
            )
        )
    return fig


@callback(
    Output('delay-categories-pie', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_delay_categories_pie(document_types, approval_stages):
    categories = delay_categories(df, document_type_filter=document_types, state_filter=approval_stages)
    labels = list(categories.keys())
    values = list(categories.values())

    if sum(values) > 0:
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            direction='clockwise',sort=True,insidetextorientation='horizontal',
            hole=0.4,
            marker_colors=colors_palette),
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True,
            height=None,
            width=None
        )
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(
            title='Delay Categories',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),

        )
        fig.add_annotation(
            text="No data available for the selected filters",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )
    return fig

@callback(
    Output('non-approval-states-bar', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_non_approval_states_bar(document_types, approval_stages):
    state_non_approval_df = states_percentage_non_approvals(
        df, document_type_filter=document_types, state_filter=approval_stages)

    if not state_non_approval_df.empty:
        state_non_approval_df = state_non_approval_df.sort_values('non_approval_percentage', ascending=False)

        fig = go.Figure(go.Bar(
            x=state_non_approval_df['non_approval_percentage'],
            y=state_non_approval_df['state'],
            orientation='h',
            marker=dict(
                color=colors_palette[1],
                line=dict(
                    color='#dcdcdc',
                    width=1
                )
            ),
        ))
        fig.update_layout(
            xaxis_title='Percentage (%)',
            yaxis_title='Approval Stage',
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                range=[0, 100],
                gridcolor='#dcdcdc'
            ),
            yaxis=dict(
                autorange="reversed",
                gridcolor='#dcdcdc'
            )
        )
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(
            title='Percentage of Non-Approvals by State',
            xaxis_title='Percentage (%)',
            yaxis_title='State',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),

        )
        fig.add_annotation(
            text="No data available for the selected filters",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )
    return fig

@callback(
    Output('loop-states-bar', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_loop_states_bar(document_types, approval_stages):
    max_iteration_counts = max_iterations_counts(
        df, document_type_filter=document_types, state_filter=approval_stages)

    if not max_iteration_counts.empty:
        max_iteration_counts = max_iteration_counts.sort_values('max_iteration')

        fig = go.Figure(go.Bar(
            x=max_iteration_counts['max_iteration'],
            y=max_iteration_counts['document_count'],
            marker=dict(
                color=colors_palette[3],
                line=dict(
                    color='#dcdcdc',
                    width=1
                )
            ),
        ))

        fig.update_layout(
            xaxis_title='Approval Loop',
            yaxis_title='Number of Documents',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                gridcolor='#dcdcdc',
                type='category'
            ),
            yaxis=dict(
                gridcolor='#dcdcdc'
            ),
            autosize=True,
            height=None,
            width=None
        )
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(
            title='Number of Documents by Maximum Number of Iterations',
            xaxis_title='Maximum Number of Iterations',
            yaxis_title='Number of Documents',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),

        )
        fig.add_annotation(
            text="No data available for the selected filters",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )
    return fig

@callback(
    Output('mean-loop-states-bar', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_mean_loop_states_bar(document_types, approval_stages):
    state_mean_iter_df = states_mean_max_iterations(
        df, document_type_filter=document_types, state_filter=approval_stages)

    if not state_mean_iter_df.empty:
        state_mean_iter_df = state_mean_iter_df.sort_values('mean_max_iteration', ascending=False)

        fig = go.Figure(go.Bar(
            x=state_mean_iter_df['mean_max_iteration'],
            y=state_mean_iter_df['state'],
            orientation='h',
            marker=dict(
                color=colors_palette[2],
                line=dict(
                    color='#dcdcdc',
                    width=1
                )
            ),
        ))
        fig.update_layout(
            xaxis_title='Mean Iteration Number',
            yaxis_title='Approval Stage',
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(
                autorange="reversed",
                gridcolor='#dcdcdc'
            ),
            xaxis=dict(
                gridcolor='#dcdcdc'
            )
        )
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Mean Iteration Number',
            yaxis_title='Approval Stage',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),

        )
        fig.add_annotation(
            text="No data available for the selected filters",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )
    return fig


@callback(
    Output('reverse-motion-gauge', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_reverse_motion_gauge(document_types, approval_stages):
    count_loop_documents, total_documents, percentage = calculate_loop_percentage(
        df, document_type_filter=document_types, state_filter=approval_stages)

    percentage = max(0, min(percentage, 100))
    gauge_max = 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        number={'valueformat': '.2f', 'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.25], 'color': 'lightgreen'},
                {'range': [gauge_max * 0.25, gauge_max * 0.5], 'color': 'yellow'},
                {'range': [gauge_max * 0.5, gauge_max * 0.75], 'color': 'orange'},
                {'range': [gauge_max * 0.75, gauge_max], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': percentage
            }
        },
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=5),
        height=100,
        template=None,
    )

    return fig


@callback(
    Output('non-approvals-gauge', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_non_approvals_gauge(document_types, approval_stages):
    percentage = percentage_non_approvals(
        df, document_type_filter=document_types, state_filter=approval_stages)

    percentage = max(0, min(percentage, 100))
    gauge_max = 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        number={'valueformat': '.2f', 'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "crimson"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.25], 'color': 'lightgreen'},
                {'range': [gauge_max * 0.25, gauge_max * 0.5], 'color': 'yellow'},
                {'range': [gauge_max * 0.5, gauge_max * 0.75], 'color': 'orange'},
                {'range': [gauge_max * 0.75, gauge_max], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': percentage
            }
        },
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=5),
        height=100,
        template=None,
    )

    return fig

@callback(
    Output('percentage-delays-gauge', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_percentage_delays_gauge(document_types, approval_stages):
    percentage = percentage_of_delays(
        df, document_type_filter=document_types, state_filter=approval_stages)

    percentage = max(0, min(percentage, 100))
    gauge_max = 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        number={'valueformat': '.2f', 'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.25], 'color': 'lightgreen'},
                {'range': [gauge_max * 0.25, gauge_max * 0.5], 'color': 'yellow'},
                {'range': [gauge_max * 0.5, gauge_max * 0.75], 'color': 'orange'},
                {'range': [gauge_max * 0.75, gauge_max], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': percentage
            }
        },
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=5),
        height=100,
        template=None,
    )

    return fig


@callback(
    Output('average-delay-gauge', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_average_delay_gauge(document_types, approval_stages):
    avg_delay, max_delay = average_and_max_delay(
        df, document_type_filter=document_types, state_filter=approval_stages)

    gauge_max = max(max_delay, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_delay,
        number={'valueformat': '.2f', 'suffix': "h", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.25], 'color': 'lightgreen'},
                {'range': [gauge_max * 0.25, gauge_max * 0.5], 'color': 'yellow'},
                {'range': [gauge_max * 0.5, gauge_max * 0.75], 'color': 'orange'},
                {'range': [gauge_max * 0.75, gauge_max], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': avg_delay
            }
        },
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=5),
        height=100,
        template=None,
    )

    return fig

@callback(
    Output('average-iterations-gauge', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_average_iterations_gauge(document_types, approval_stages):
    avg_iterations, max_iterations = average_and_max_iterations(
        df, document_type_filter=document_types, state_filter=approval_stages)

    gauge_max = max(max_iterations, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_iterations,
        number={'valueformat': '.2f', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "mediumseagreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.25], 'color': 'lightgreen'},
                {'range': [gauge_max * 0.25, gauge_max * 0.5], 'color': 'yellow'},
                {'range': [gauge_max * 0.5, gauge_max * 0.75], 'color': 'orange'},
                {'range': [gauge_max * 0.75, gauge_max], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': avg_iterations
            }
        },
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=5),
        height=100,
        template=None,
    )

    return fig

@callback(
    Output('approval-result-pie-chart', 'figure'),
    [Input('document-type-dropdown', 'value'),
     Input('approval-stage-dropdown', 'value')]
)
def update_approval_result_pie_chart(document_types, approval_stages):
    distribution_df = approval_result_distribution(
        df, document_type_filter=document_types, state_filter=approval_stages)

    if not distribution_df.empty:
        labels = distribution_df['approval_result']
        values = distribution_df['count']

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            direction='clockwise',
            sort=True,
            insidetextorientation='horizontal',
            hole=0.4,
            marker=dict(colors=colors_palette),
        ))

        fig.update_traces(textposition='inside', textinfo='percent+label')

        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True,
            height=None,
            width=None
        )
        return fig
    else:
        fig = px.pie()
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),

        )
        fig.add_annotation(
            text="No data available for the selected filters",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
    return fig