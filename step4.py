import dash
from dash import html, dcc, Input, Output, State, callback, ctx, dash_table, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pyomo.environ as pyo
from functools import reduce


def run_optimization_ipopt(model, x0, variable_transition_indices,
                           m_variable_indices, state_indicies,
                           initial_m_values, fixed_m_states,
                           opt_parameter, function_name, objective_function):
    if opt_parameter == 'max':
        sense = pyo.maximize
    else:
        sense = pyo.minimize
    model.Objective = pyo.Objective(rule=objective_function, sense=sense)

    for (from_state, to_state), idx in variable_transition_indices.items():
        model.q[from_state, to_state].value = x0[idx]

    for state, idx in m_variable_indices.items():
        model.m[state].value = x0[idx]

    initial_q_values = {}
    for (from_state, to_state) in model.VariableTransitions:
        initial_q_values[(from_state, to_state)] = model.q[from_state, to_state].value

    initial_m_values_model = {}
    for state in model.VariableMStates:
        initial_m_values_model[state] = model.m[state].value

    solver = pyo.SolverFactory('ipopt')
    solver.options['halt_on_ampl_error'] = 'yes'
    result = solver.solve(model, tee=True)
    optimized_m = {}
    for state in model.VariableMStates:
        optimized_m[state] = model.m[state].value
    for state in model.FixedMStates:
        optimized_m[state] = model.InitialMValues[state]

    optimized_q = {}
    for (from_state, to_state) in model.VariableTransitions:
        optimized_q[(from_state, to_state)] = model.q[from_state, to_state].value
    for (from_state, to_state) in model.FixedTransitions:
        optimized_q[(from_state, to_state)] = model.Intensities[from_state, to_state]

    control_parameters = []
    values_before_optimization = []
    values_after_optimization = []

    # Collect data for variable transitions
    for (from_state, to_state), idx in variable_transition_indices.items():
        param_name = f"int[{state_indicies[from_state]} -> {state_indicies[to_state]}]"
        initial_value = initial_q_values[(from_state, to_state)]
        optimized_value = optimized_q[(from_state, to_state)]
        control_parameters.append(param_name)
        values_before_optimization.append(round(initial_value, 3))
        values_after_optimization.append(round(optimized_value, 3))

    for state, idx in m_variable_indices.items():
        param_name = f"m[{state_indicies[state]}]"
        initial_value = initial_m_values_model[state]
        optimized_value = optimized_m[state]
        control_parameters.append(param_name)
        values_before_optimization.append(round(initial_value))
        values_after_optimization.append(round(optimized_value))

    for state in fixed_m_states:
        param_name = f"m[{state_indicies[state]}] (fixed)"
        value = initial_m_values[state]
        control_parameters.append(param_name)
        values_before_optimization.append(round(value))
        values_after_optimization.append(round(value))

    results_df = pd.DataFrame({
        'Control Parameters': control_parameters,
        'Values Before Optimization': values_before_optimization,
        f'Values After Optimization ({function_name})': values_after_optimization
    })

    objective_value = pyo.value(model.Objective)

    original_values = {}
    for var in model.component_objects(pyo.Var, descend_into=True):
        for index in var:
            original_values[(var.name, index)] = var[index].value

    for (from_state, to_state), idx in variable_transition_indices.items():
        model.q[from_state, to_state].value = initial_q_values[(from_state, to_state)]

    for state, idx in m_variable_indices.items():
        model.m[state].value = initial_m_values_model[state]

    initial_objective_value = pyo.value(model.Objective)

    for (from_state, to_state), idx in variable_transition_indices.items():
        model.q[from_state, to_state].value = optimized_q[(from_state, to_state)]

    for state, idx in m_variable_indices.items():
        model.m[state].value = optimized_m[state]

    obj_df = pd.DataFrame({
        'Before Optimization': [round(initial_objective_value, 2)],
        f'({function_name})': [round(objective_value, 2)]
    })

    return results_df, obj_df

layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4([
                "Optimization Results",
                html.I(className="fas fa-question-circle", id="tooltip-target-opt-results",
                       style={'color': 'gray', 'marginLeft': '10px', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                "This tooltip provides additional information about the intensity matrix.",
                target="tooltip-target-opt-results",
                placement="right",
            )])]),
dbc.Button('Run Optimization', id='run-button', n_clicks=0, color='primary'),
    dbc.Row(dbc.Col(html.Div(id='results-table',style={'margin-top':'15px'})))
], style={'height':'70vh','overflow':'scroll'})

@callback(
    Output('results-table', 'children'),
    Input('run-button','n_clicks'),
    State('strategies-store', 'data'),
    State('intensity-matrix', 'data'),
    State('stationary-distribution', 'data'),
    State('productive-states', 'data'),
    State('unproductive-states', 'data'),
    State('state-costs', 'data'),
    State('state-constraints', 'data'),
    State('states', 'data'),
    State('N', 'data'),
    State('intensity-constraints','data'),
    prevent_initial_call=True
)
def get_optimization_results(btn,strategies_store_data, intensity_matrix_data,
                        stationary_distribution_data, productive_states_data,
                        unproductive_states_data, state_costs_data, state_constraints_data,states_stored,N_stored,
                             intensity_constraints_data):
    if btn <= 0 or not intensity_matrix_data or not stationary_distribution_data or not productive_states_data \
            or not unproductive_states_data or not state_costs_data or not strategies_store_data:
        return dash.no_update
    N = N_stored
    states = list(states_stored.keys())
    num_states = len(states)
    state_indicies = {item: int(''.join(filter(str.isdigit, item))) for item in states}

    intensity_matrix = pd.DataFrame(data=list(map(lambda d: {k: v for k, v in d.items() if k != 'State'}, intensity_matrix_data)),index=states,columns=states)
    state_costs_dict = {item['state']: item['cost'] for item in state_costs_data}
    items = [html.Li(f"{key}: {value}") for key, value in states_stored.items()]

    all_results = []
    states_div = html.Div(html.Ul(items),style={'max-height':'15vh','overflow-y':'auto','margin-bottom':'15px'})
    all_results.append(states_div)

    pivot_data = []
    before_opt_added = False
    for strategy in strategies_store_data:
        if strategy['is_saved']:
            strategy_states = strategy['states']

            all_transitions = [
                (from_state, to_state)
                for from_state in states
                for to_state in states
                if from_state != to_state and intensity_matrix.loc[from_state, to_state] != 0
            ]

            fixed_transitions = [
                (from_state, to_state)
                for from_state in states
                for to_state in strategy_states
                if from_state != to_state and intensity_matrix.loc[from_state, to_state] != 0
            ]
            variable_transitions = [t for t in all_transitions if t not in fixed_transitions]
            intensities = {}
            for (from_state, to_state) in all_transitions:
                intensity = intensity_matrix.loc[from_state, to_state]
                intensities[(from_state, to_state)] = intensity

            num_variable_intensities = len(variable_transitions)
            m_initial_data = pd.DataFrame(stationary_distribution_data)
            m_initial_data.set_index('state', inplace=True)
            initial_m_values = {state: m_initial_data.loc[state, 'm_initial'] for state in states}
            fixed_m_states = strategy_states
            variable_m_states = [state for state in states if state not in fixed_m_states]
            variable_m_initial = [initial_m_values[state] for state in variable_m_states]
            variable_intensities_initial = [intensities[trans] for trans in variable_transitions]

            x0 = np.concatenate((variable_intensities_initial, variable_m_initial))

            variable_transition_indices = {(from_state, to_state): idx for idx, (from_state, to_state) in
                                           enumerate(variable_transitions)}
            m_variable_indices = {state: idx + num_variable_intensities for idx, state in enumerate(variable_m_states)}
            epsilon = 1e-3

            model = pyo.ConcreteModel()

            model.States = pyo.Set(initialize=states)
            model.VariableTransitions = pyo.Set(initialize=variable_transitions, dimen=2)
            model.FixedTransitions = pyo.Set(initialize=fixed_transitions, dimen=2)
            model.VariableMStates = pyo.Set(initialize=variable_m_states)
            model.FixedMStates = pyo.Set(initialize=fixed_m_states)
            model.AllTransitions = model.VariableTransitions | model.FixedTransitions

            model.N = N
            model.Intensities = pyo.Param(model.AllTransitions, initialize=intensities)
            model.InitialMValues = pyo.Param(model.FixedMStates, initialize={
                state: initial_m_values[state] for state in fixed_m_states
            })
            model.StateCosts = pyo.Param(model.States, initialize=state_costs_dict)

            model.q = pyo.Var(model.VariableTransitions, bounds=(0+0.00001, 0.2 + 0.00001))
            model.m = pyo.Var(model.VariableMStates, bounds=(epsilon, N - epsilon))

            if state_constraints_data:
                for constraint in state_constraints_data:
                    state = constraint['state']
                    if state in variable_m_states:
                        min_limit = constraint['min'] * N * 0.01 + epsilon
                        max_limit = constraint['max'] * N * 0.01 - epsilon
                        model.m[state].setlb(min_limit)
                        model.m[state].setub(max_limit)

            if intensity_constraints_data:
                for constraint in intensity_constraints_data:
                    from_state = constraint['from_state']
                    to_state = constraint['to_state']
                    if (from_state, to_state) in model.VariableTransitions:
                        min_limit = constraint['min']
                        max_limit = constraint['max']
                        model.q[from_state, to_state].setlb(min_limit)
                        model.q[from_state, to_state].setub(max_limit)

            def get_q(model, from_state, to_state):
                if (from_state, to_state) in model.VariableTransitions:
                    return model.q[from_state, to_state]
                elif (from_state, to_state) in model.FixedTransitions:
                    return model.Intensities[from_state, to_state]
                else:
                    return 0.0

            def stationarity_rule(model, j):
                m_total = {}
                for i in model.States:
                    if i in model.VariableMStates:
                        m_i = model.m[i]
                    else:
                        m_i = model.InitialMValues[i]
                    m_total[i] = m_i

                incoming = sum(m_total[i] * get_q(model, i, j) for i in model.States if i != j)
                outgoing = m_total[j] * sum(get_q(model, j, k) for k in model.States if k != j)
                return incoming - outgoing == 0

            model.StationarityConstraints = pyo.Constraint(model.States, rule=stationarity_rule)

            def total_elements_rule(model):
                total_m = sum(model.m[state] for state in model.VariableMStates) + \
                          sum(model.InitialMValues[state] for state in model.FixedMStates)
                return total_m == model.N

            model.TotalElementsConstraint = pyo.Constraint(rule=total_elements_rule)

            def fun1(model):
                total_cost = sum(
                    (model.m[state] if state in model.VariableMStates else model.InitialMValues[state]) *
                    model.StateCosts[state]
                    for state in productive_states_data
                )
                return total_cost

            def fun2(model):
                total = 0
                for state in unproductive_states_data:
                    if state in model.VariableMStates:
                        m_state = model.m[state]
                    elif state in model.FixedMStates:
                        m_state = model.InitialMValues[state]

                    expr = m_state * (1 - (m_state / model.N))

                    total += model.StateCosts[state] * pyo.sqrt(expr + epsilon)
                return total

            def fun3(model):
                total = 0
                for state in productive_states_data:
                    if state in model.VariableMStates:
                        m_state = model.m[state]
                    elif state in model.FixedMStates:
                        m_state = model.InitialMValues[state]

                    expr = m_state * (1 - (m_state / model.N))

                    total += model.StateCosts[state] * pyo.sqrt(expr + epsilon)
                return total

            def fun4(model):
                total_unproductive = fun2(model)
                total_productive = fun1(model)
                return total_productive - total_unproductive
            def fun5(model):
                total_unproductive = fun3(model)
                total_productive = fun1(model)
                return total_productive - total_unproductive

            objectives = [
                ('F1', fun1, 'max'),
                ('F2', fun2, 'min'),
                ('F3', fun3, 'min'),
                ('F4', fun4, 'max'),
                ('F5', fun5, 'max')
            ]

            obj_before_dict = {}
            obj_after_dict = {}
            results_dfs = []
            for function_name, objective_function, opt_parameter in objectives:
                results_df, obj_df = run_optimization_ipopt(
                    model=model,
                    x0=x0,
                    variable_transition_indices=variable_transition_indices,
                    m_variable_indices=m_variable_indices,
                    state_indicies=state_indicies,
                    initial_m_values=initial_m_values,
                    fixed_m_states=fixed_m_states,
                    opt_parameter=opt_parameter,
                    function_name=function_name,
                    objective_function=objective_function
                )
                obj_before_dict[function_name] = obj_df['Before Optimization'].iloc[0]
                obj_after_dict[function_name] = obj_df[f'({function_name})'].iloc[0]
                df_to_merge = results_df[['Control Parameters', 'Values Before Optimization',
                                          f'Values After Optimization ({function_name})']]
                results_dfs.append(df_to_merge)

            if results_dfs:
                merged_df = reduce(
                    lambda left, right: pd.merge(
                        left, right, on=['Control Parameters', 'Values Before Optimization'], how='outer'
                    ),
                    results_dfs
                )
            else:
                merged_df = pd.DataFrame()

            if not before_opt_added:
                before_opt_added = True
                obj_before_dict['Strategy'] = 'Before Optimization'
                pivot_data.append(obj_before_dict.copy())

            obj_after_dict['Strategy'] = f"Strategy {strategy['number']}"
            pivot_data.append(obj_after_dict.copy())

            strategy_header = html.H4(f"Results for Strategy {strategy['number']}")
            table = dbc.Table.from_dataframe(merged_df, striped=True, bordered=True, hover=True)
            table_div = html.Div(table,style={'max-height':'30vh','overflow-y':'auto'})
            all_results.append(strategy_header)
            all_results.append(table_div)
            all_results.append(html.Hr())

    pivot_header = html.H4(f"Pivot Table for Strategies")
    pivot_df = pd.DataFrame(pivot_data)
    pivot_table = dbc.Table.from_dataframe(pivot_df, striped=True, bordered=True, hover=True)
    pivot_table_div = html.Div(pivot_table, style={'max-height': '30vh', 'overflow-y': 'auto'})
    all_results.append(pivot_header)
    all_results.append(pivot_table_div)
    return all_results



