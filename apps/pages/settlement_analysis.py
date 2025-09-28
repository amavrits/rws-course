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
    """Create a Plotly figure from predictions data."""
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
        # Fallback for older data without target probabilities
        prior_exceeds_target = [0] * len(forecast_times)
        posterior_exceeds_target = [0] * len(forecast_times)
        has_target_data = False
    
    # Calculate true settlement for display
    from src.settlement.model import settlement_model, SoilParams
    params = SoilParams()
    true_settlement = settlement_model(times=np.array(all_times), params=params, cv=np.array([true_cv]))
    
    # Create subplot figure with 2 columns and secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Settlement Predictions', 'Cv Distribution'),
        column_widths=[0.75, 0.25],
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Left plot - Settlement predictions
    # Observations
    if obs_times:
        fig.add_trace(
            go.Scatter(
                x=obs_times,
                y=settlement_obs,
                mode='markers',
                marker=dict(color='black', symbol='x', size=8),
                name='Observations',
                legendgroup='left'
            ),
            row=1, col=1
        )
    
    # Prior predictions - upper bound (invisible for fill)
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=prior_upper_quantile,
            mode='lines',
            line=dict(width=0.5, color='blue'),
            showlegend=False,
            hoverinfo='skip',
            legendgroup='left'
        ),
        row=1, col=1
    )
    
    # Prior predictions - lower bound with fill
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=prior_lower_quantile,
            mode='lines',
            line=dict(width=0.5, color='blue'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.3)',
            name='Prior 90% CI',
            hoverinfo='skip',
            legendgroup='left'
        ),
        row=1, col=1
    )
    
    # Prior mean
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=prior_mean,
            mode='lines',
            line=dict(color='blue', width=1.5),
            name='Prior mean prediction',
            legendgroup='left'
        ),
        row=1, col=1
    )
    
    # Posterior predictions - upper bound (invisible for fill)
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=posterior_upper_quantile,
            mode='lines',
            line=dict(width=0.5, color='red'),
            showlegend=False,
            hoverinfo='skip',
            legendgroup='left'
        ),
        row=1, col=1
    )
    
    # Posterior predictions - lower bound with fill
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=posterior_lower_quantile,
            mode='lines',
            line=dict(width=0.5, color='red'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            name='Posterior 90% CI',
            hoverinfo='skip',
            legendgroup='left'
        ),
        row=1, col=1
    )
    
    # Posterior mean
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=posterior_mean,
            mode='lines',
            line=dict(color='red', width=1.5),
            name='Posterior mean prediction',
            legendgroup='left'
        ),
        row=1, col=1
    )
    
    # True settlement
    fig.add_trace(
        go.Scatter(
            x=all_times,
            y=true_settlement.flatten(),
            mode='lines',
            line=dict(color='green', width=1.5),
            name='True settlement model',
            showlegend=False  # Will be handled by custom legend
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Secondary y-axis traces (target exceedance probabilities)
    if has_target_data:
        fig.add_trace(
            go.Scatter(
                x=forecast_times,
                y=prior_exceeds_target,
                mode='lines',
                line=dict(color='blue', dash='dot'),
                name='P(S>Target) - Prior',
                showlegend=False  # Will be handled by custom legend
            ),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_times,
                y=posterior_exceeds_target,
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='P(S>Target) - Posterior',
                showlegend=False  # Will be handled by custom legend
            ),
            row=1, col=1, secondary_y=True
        )
    
    # Vertical line at observation time
    if obs_times:
        fig.add_vline(
            x=max(obs_times),
            line=dict(color='black', dash='dash'),
            row=1, col=1
        )
    
    # Right plot - Cv distributions (matching matplotlib orientation)
    fig.add_trace(
        go.Scatter(
            x=cv_grid,
            y=cv_prior_pdf,
            mode='lines',
            line=dict(color='blue'),
            name='Prior PDF',
            legendgroup='right'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cv_grid,
            y=cv_posterior_pdf,
            mode='lines',
            line=dict(color='red'),
            name='Posterior PDF',
            legendgroup='right'
        ),
        row=1, col=2
    )
    
    # True Cv line
    fig.add_vline(
        x=true_cv,
        line=dict(color='green', width=2),
        row=1, col=2
    )
    
    # Add invisible trace for True Cv legend entry
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='green', width=2),
            name='True Cv',
            legendgroup='right'
        ),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time [d]", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Settlement [m]", autorange='reversed', row=1, col=1, secondary_y=False, showgrid=True)
    if has_target_data:
        fig.update_yaxes(title_text="Target exceedance probability [%]", range=[0, 100], row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Cv [m²/d]", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="Density [-]", row=1, col=2, showgrid=True)
    
    fig.update_layout(
        height=650,  # Increased height for legend above
        showlegend=False,  # Disable default legend, use custom annotations
        title=f"Settlement Analysis - Time: {time_key} days"
    )
    
    # Create custom legend above left subplot in 4 columns
    left_legend_items = [
        ("Observations", "black", "markers"),
        ("Prior mean prediction", "blue", "line"),
        ("Prior 90% CI", "rgba(0,0,255,0.3)", "fill"),
        ("True settlement model", "green", "line")
    ]
    
    if has_target_data:
        left_legend_items.extend([
            ("Posterior mean prediction", "red", "line"), 
            ("Posterior 90% CI", "rgba(255,0,0,0.3)", "fill"),
            ("P(S>Target) - Prior", "blue", "dash"),
            ("P(S>Target) - Posterior", "red", "dash")
        ])
    else:
        left_legend_items.extend([
            ("Posterior mean prediction", "red", "line"), 
            ("Posterior 90% CI", "rgba(255,0,0,0.3)", "fill")
        ])
    
    # Create 4-column legend above left subplot
    legend_html = "<b>Settlement Analysis Legend</b><br>"
    legend_html += "<table style='font-size: 12px; border-spacing: 10px 2px;'>"
    
    for i in range(0, len(left_legend_items), 4):  # Process 4 items per row
        legend_html += "<tr>"
        for j in range(4):
            if i + j < len(left_legend_items):
                name, color, style = left_legend_items[i + j]
                
                if style == "line":
                    symbol = "—"
                elif style == "dash": 
                    symbol = "- - -"
                elif style == "markers":
                    symbol = "✕"
                elif style == "fill":
                    symbol = "█"
                else:
                    symbol = "—"
                    
                legend_html += f"<td><span style='color:{color}'>{symbol}</span> {name}</td>"
            else:
                legend_html += "<td></td>"  # Empty cell
        legend_html += "</tr>"
    
    legend_html += "</table>"
    
    fig.add_annotation(
        x=0.375,  # Center of left subplot
        y=1.15,   # Above the subplot
        xref='paper',
        yref='paper',
        text=legend_html,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="Black",
        borderwidth=1,
        align="center",
        xanchor="center"
    )
    
    # Create custom legend for right subplot
    right_legend_items = []
    for trace in fig.data:
        if hasattr(trace, 'legendgroup') and trace.legendgroup == 'right' and trace.showlegend is not False:
            # Hide from main legend
            trace.showlegend = False
            # Collect for custom legend
            right_legend_items.append(f"<span style='color:{trace.line.color}'>{trace.name}</span>")
    
    # Add right legend as annotation (right top corner of right plot)
    if right_legend_items:
        legend_text = "<br>".join(right_legend_items)
        fig.add_annotation(
            x=0.98,  # Right edge of right subplot
            y=0.95,  # Top of subplot
            xref='paper',
            yref='paper',
            text=legend_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="Black",
            borderwidth=1,
            align="left",
            xanchor="right"
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
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        html.Div([
            dcc.Graph(id='settlement-analysis-plot', style={'height': '650px'})  # Increased height for legend
        ], style={'margin': '20px'}),
        
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
                html.Li("The right panel shows how the Cv parameter distribution evolves")
            ])
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'})
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