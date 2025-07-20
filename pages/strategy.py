from dash import html, dcc, Input, Output, State, callback
import dash_mantine_components as dmc
import dash
import dash_bootstrap_components as dbc
import step1
import step2
import step3
import step4

dash.register_page(__name__, path='/strategy')

TOTAL_STEPS = 4


def get_step_content(step):
    if step == 1:
        return step1.layout
    elif step == 2:
        return step2.layout
    elif step == 3:
        return step3.layout
    elif step == 4:
        return step4.layout

descriptions = ['Modelling','Optimization Criteria and Constraints','Strategies','Optimization Results']

def get_stepper(current_step):
    items = []
    for i in range(TOTAL_STEPS):
        items.append(
            dmc.StepperStep(
                label=f"Step {i + 1}",
                description=descriptions[i]
            )
        )
    return dmc.Stepper(
        active=current_step - 1,
        children=items,
        size="sm",
        styles={
            'stepIcon': {'borderColor': 'blue'},
            'completedStepIcon': {'borderColor': 'green'},
            'stepBody': {'textAlign': 'center'}
        }
    )

layout = html.Div([
    html.H1('Management Strategy',style={'margin-bottom':'15px'}),
    dcc.Store(id='step-store', data=1),
    html.Div(id='stepper'),
    html.Br(),
    html.Div(id='step-content'),
    html.Br(),
    dbc.Button('Previous', id='prev-btn', color='secondary', className='me-2'),
    dbc.Button('Next', id='next-btn', color='primary'),

],style={'margin':'15px 30px'})




@callback(
    Output('step-store', 'data'),
    [Input('next-btn', 'n_clicks'),
     Input('prev-btn', 'n_clicks')],
    State('step-store', 'data'),
    prevent_initial_call=True
)
def update_step(n_next, n_prev, current_step):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_step
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'next-btn' and current_step < TOTAL_STEPS:
        return current_step + 1
    elif button_id == 'prev-btn' and current_step > 1:
        return current_step - 1
    return current_step

@callback(
    [Output('step-content', 'children'),
     Output('stepper', 'children'),
     Output('prev-btn', 'disabled'),
     Output('next-btn', 'children'),
     Output('next-btn', 'disabled')],
    Input('step-store', 'data')
)
def display_step(step):
    prev_disabled = step == 1
    next_label = 'Finish' if step == TOTAL_STEPS else 'Next'
    next_disabled = False
    return (
        get_step_content(step),
        get_stepper(step),
        prev_disabled,
        next_label,
        next_disabled
    )