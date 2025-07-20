import dash
from dash import html, dcc, dash_table
from dash import Input, Output, State, callback
import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
import sqlite3
import utils

dash.register_page(__name__, path='/loading')



layout = html.Div([
    html.Div([
        html.H4([
            "Loading Data",
            html.I(
                className="fas fa-question-circle",
                id="tooltip-target-data-loading",
                style={
                    'color': 'gray',
                    'marginLeft': '10px',
                    'cursor': 'pointer'
                }
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        dbc.Tooltip(
            "This tooltip provides additional information about the intensity matrix.",
            target="tooltip-target-data-loading",
            placement="right",
        )
    ]),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select an Excel File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            accept='.xlsx'
        ),
    ]),
    html.Div(
        id='output-data-upload',
        style={
            'flex-grow': '1',
            'overflow': 'auto'
        }
    )
],
style={
    'display': 'flex',
    'flexDirection': 'column',
    'height': '100vh',
    'margin': '15px 30px'
})

@callback(
    Output('data-has-been-loaded', 'data'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),prevent_initial_call=True
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_excel(io.BytesIO(decoded))
            conn = sqlite3.connect('application.db')
            df.to_sql('df_table', conn, if_exists='replace', index=False)
            if not df.empty:
                data_has_been_loaded = True
                div = html.Div([
                html.H5(f"{filename} has been saved to the database."),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns]
                )
            ])
                return data_has_been_loaded, div
            else:
                return dash.no_update, dash.no_update
        except Exception as e:
            div = html.Div([
                'There was an error processing this file.'
            ])
            return dash.no_update, div
    else:
        return dash.no_update, dash.no_update