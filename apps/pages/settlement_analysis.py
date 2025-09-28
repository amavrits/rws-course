"""
Settlement Analysis Page for Multi-Page Dash App
===============================================
This page provides settlement analysis visualization integrated into the main app.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots

# Register this page
dash.register_page(__name__, path="/settlement-analysis", name="Settlement Analysis")

def load_predictions_data():
    """Load predictions data from the JSON file."""
    data_path = Path(__file__).parent.parent.parent / "data/settlement_analysis/predictions.json"
    with open(data_path, "r") as f:
        predictions = json.load(f)
    return predictions

def create_plotly_figure(predictions_data, time_key, true_cv=2.3*1e-8 * (24 * 3_600)):
    """Create a clean Plotly figure from predictions data."""
    prediction = predictions_data[str(time_key)]

    all_times = prediction["all_times"]
    obs_times = prediction["observation_times"]
    forecast_times = prediction["forecast_times"]
    settlement_obs = prediction["observations"]
    prior_mean = prediction["prior_mean"]
    prior_lower_quantile = prediction["prior_lower_quantile"]
    prior_upper_quantile = prediction["prior_upper_quantile"]
    posterior_mean = prediction["posterior_mean"]
    posterior_lower_quantile = prediction["posterior_lower_quantile"]
    posterior_upper_quantile = prediction["posterior_upper_quantile"]
    cv_grid = prediction["cv_grid"]
    cv_prior_pdf = prediction["cv_prior_pdf"]
    cv_posterior_pdf = prediction["cv_posterior_pdf"]

    # Get target exceedance probabilities (convert to percentages)
    try:
        prior_exceeds_target = [p*100 for p in prediction["prior_exceeds_target"]]
        posterior_exceeds_target = [p*100 for p in prediction["posterior_exceeds_target"]]
        has_target_data = True
    except KeyError:
        prior_exceeds_target = [0] * len(forecast_times)
        posterior_exceeds_target = [0] * len(forecast_times)
        has_target_data = False

    # Calculate true settlement for display
    from src.settlement.model import settlement_model, SoilParams
    params = SoilParams()
    true_settlement = settlement_model(times=np.array(all_times), params=params, cv=np.array([true_cv]))
    target_settlement = 0.067

    # Create subplot figure with 2 columns and secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Settlement Predictions", "Cv Distribution"),
        column_widths=[0.8, 0.2],  # Give more space to main plot
        horizontal_spacing=0.1,     # Reduce spacing to maximize plot area
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )

    # LEFT PLOT - Settlement predictions
    # Observations
    if obs_times:
        fig.add_trace(
            go.Scatter(
                x=obs_times, y=settlement_obs,
                mode="markers", marker=dict(color="black", symbol="x", size=10),
                name="Observations", showlegend=True
            ), row=1, col=1, secondary_y=False
        )
        # Add vertical line at observation time
        fig.add_vline(x=max(obs_times), line=dict(color="black", dash="dash"), row=1, col=1)

    # Prior confidence interval (fill area)
    fig.add_trace(
        go.Scatter(
            x=forecast_times + forecast_times[::-1],
            y=prior_upper_quantile + prior_lower_quantile[::-1],
            fill="toself", fillcolor="rgba(0,0,255,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prior 90% CI", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Prior mean
    fig.add_trace(
        go.Scatter(
            x=forecast_times, y=prior_mean,
            mode="lines", line=dict(color="blue", width=2),
            name="Prior mean", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Posterior confidence interval (fill area)
    fig.add_trace(
        go.Scatter(
            x=forecast_times + forecast_times[::-1],
            y=posterior_upper_quantile + posterior_lower_quantile[::-1],
            fill="toself", fillcolor="rgba(255,0,0,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Posterior 90% CI", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Posterior mean
    fig.add_trace(
        go.Scatter(
            x=forecast_times, y=posterior_mean,
            mode="lines", line=dict(color="red", width=2),
            name="Posterior mean", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # True settlement
    fig.add_trace(
        go.Scatter(
            x=all_times, y=true_settlement.flatten(),
            mode="lines", line=dict(color="green", width=2),
            name="True settlement", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Target settlement line
    fig.add_hline(y=target_settlement, line=dict(color="black", width=2), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="black", width=2),
            name="Target settlement", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Target exceedance probabilities on secondary y-axis
    if has_target_data:
        fig.add_trace(
            go.Scatter(
                x=forecast_times, y=prior_exceeds_target,
                mode="lines", line=dict(color="blue", dash="dot", width=2),
                name="P(S>Target) Prior", yaxis="y2", showlegend=True
            ), row=1, col=1, secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_times, y=posterior_exceeds_target,
                mode="lines", line=dict(color="red", dash="dot", width=2),
                name="P(S>Target) Posterior", yaxis="y2", showlegend=True
            ), row=1, col=1, secondary_y=True
        )

    # RIGHT PLOT - Cv distributions (don't show in main legend)
    fig.add_trace(
        go.Scatter(
            x=cv_grid, y=cv_prior_pdf,
            mode="lines", line=dict(color="blue", width=2),
            name="Prior PDF", showlegend=False
        ), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=cv_grid, y=cv_posterior_pdf,
            mode="lines", line=dict(color="red", width=2),
            name="Posterior PDF", showlegend=False
        ), row=1, col=2
    )

    # True Cv vertical line
    fig.add_vline(x=true_cv, line=dict(color="green", width=2), row=1, col=2)

    # Update axes
    fig.update_xaxes(title_text="Time [d]", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Settlement [m]", autorange="reversed", row=1, col=1, secondary_y=False, showgrid=True)
    if has_target_data:
        fig.update_yaxes(title_text="Target exceedance probability [%]", range=[0, 100], row=1, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Cv [m²/d]", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="Density [-]", row=1, col=2, showgrid=True)

    # Update layout with built-in legend positioned at top center, configured to wrap
    fig.update_layout(
        height=700,  # Increased height to match container
        title=f"Settlement Analysis - Time: {time_key} days",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.2,
            xanchor="center",
            y=-0.35,  # Adjusted for larger height
            yanchor="bottom",
            itemwidth=140,
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10),
        )
    )

    fig.update_yaxes(range=[0, 0.1], row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 100], row=1, col=1, secondary_y=True)

    entries = [
        dict(color="blue", label="Prior PDF"),
        dict(color="red", label="Posterior PDF"),
        dict(color="green", label="True ${C}_{v}$"),
    ]

    y0 = 0.96
    dy = 0.08
    x_line0, x_line1 = 0.45, 0.60
    x_text = 0.63

    for i, e in enumerate(entries):
        y = y0 - i * dy
        # line sample
        fig.add_shape(
            type="line",
            x0=x_line0, y0=y, x1=x_line1, y1=y,
            xref="x2 domain", yref="y2 domain",
            line=dict(color=e["color"], width=3)
        )
        # label
        fig.add_annotation(
            x=x_text, y=y, xref="x2 domain", yref="y2 domain",
            text=e["label"], showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=3
        )

    return fig

# Load predictions data
try:
    predictions_data = load_predictions_data()
    available_times = sorted([int(k) for k in predictions_data.keys()])
    data_loaded = True
except Exception as e:
    predictions_data = {}
    available_times = []
    data_loaded = False
    error_message = str(e)

# Page layout
def layout():
    if not data_loaded:
        return html.Div([
            html.H1("Settlement Analysis", style={'textAlign': 'center', 'color': '#dc3545'}),
            html.Div([
                html.H4("Data Loading Error"),
                html.P(f"Could not load settlement analysis data: {error_message}"),
                html.P("Please ensure the settlement analysis has been run first by executing:"),
                html.Code("python main/run_settlement_analysis.py", 
                         style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'display': 'block'})
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8d7da', 
                     'borderRadius': '5px', 'border': '1px solid #f5c6cb'})
        ])
    
    return html.Div([
        html.H1("Settlement Analysis Interactive Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        html.Div([
            html.Label("Select Examination Time (days):", 
                      style={'fontWeight': 'bold', 'marginBottom': 10}),
            dcc.Slider(
                id='settlement-time-slider',
                min=min(available_times),
                max=max(available_times),
                value=min(available_times),
                marks={time: {'label': str(time), 'style': {'fontSize': '10px'}} 
                       for i, time in enumerate(available_times) if i % 2 == 0},
                step=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'margin': '10px 5px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        html.Div([
            dcc.Graph(id='settlement-analysis-plot', style={'height': '700px', 'width': '100%'})
        ], style={'margin': '10px 5px', 'width': '98%'}),
        
        html.Div([
            html.H4("About this Analysis:"),
            html.P([
                "This dashboard shows the evolution of settlement predictions over time. ",
                "The settlement analysis uses Bayesian updating to improve predictions as more observations become available."
            ]),
            html.Ul([
                html.Li("Blue lines show prior predictions (before incorporating observations)"),
                html.Li("Red lines show posterior predictions (after incorporating observations)"),
                html.Li("Green line shows the true settlement model"),
                html.Li("Black crosses show actual observations"),
                html.Li("Dotted lines show target exceedance probabilities (if available)"),
                html.Li("The right panel shows how the ${C}_{v}$ parameter distribution evolves")
            ])
        ], style={'margin': '10px 5px', 'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'})
    ])

# Callbacks
@callback(
    Output('settlement-time-slider', 'marks'),
    Input('settlement-time-slider', 'value')
)
def update_slider_marks(selected_time):
    """Update slider marks to show all available times."""
    if not data_loaded:
        return {}
    
    marks = {}
    for i, time in enumerate(available_times):
        if i % max(1, len(available_times) // 10) == 0 or time == selected_time:
            marks[time] = {'label': str(time), 'style': {'fontSize': '10px'}}
    return marks

@callback(
    Output('settlement-analysis-plot', 'figure'),
    Input('settlement-time-slider', 'value')
)
def update_plot(selected_time):
    """Update the plot based on selected time."""
    if not data_loaded:
        # Return empty figure
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    # Find the closest available time
    closest_time = min(available_times, key=lambda x: abs(x - selected_time))
    
    # Create the figure
    fig = create_plotly_figure(predictions_data, closest_time)
    
    return fig