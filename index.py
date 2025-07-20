import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from app import app
from dash import dash, html,_dash_renderer
from dash import dcc
_dash_renderer._set_react_version("18.2.0")

app.layout = dmc.MantineProvider(
    children=[
        dbc.Container([
            dbc.NavbarSimple(
                brand="ISRM: Information Security Risk Management",
                brand_href="/monitoring",
                color="primary",
                dark=True,
                children=[
                    dbc.NavItem(dbc.NavLink("Strategy", href="/strategy")),
                    dbc.NavItem(dbc.NavLink("Monitoring", href="/monitoring")),
                    dbc.NavItem(dbc.NavLink("Data Loading", href="/loading")),
                ],
            ),
            dash.page_container,
        ], fluid=True)
    ],

)

app.layout.children.append(dcc.Store(id='data-has-been-loaded', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='strategies-store', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='intensity-matrix', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='states', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='N', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='stationary-distribution', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='productive-states', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='unproductive-states', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='state-costs', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='state-constraints', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='intensity-constraints', data=[],storage_type='session'))
app.layout.children.append(dcc.Store(id='doc-type-val', data=[],storage_type='session'))
if __name__ == "__main__":
    app.run(debug=False)