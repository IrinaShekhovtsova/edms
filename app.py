# app.py
from dash import Dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.15.3/css/all.css'],suppress_callback_exceptions=True)
server = app.server