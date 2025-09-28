"""
Normal Distribution Fitting Analysis
===================================
Interactive app for normal distribution fitting with different numbers of samples.
Based on run_normal_lr_inference.py functionality.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io

# Register page only if dash pages is enabled
try:
    dash.register_page(__name__, path='/normal-fitting', title='Normal Distribution Fitting')
except:
    pass

def generate_normal_data(n_total: int = 100000, true_mean: float = 10.0, true_std: float = 2.0, seed: int = 42):
    """Generate normal distribution data."""
    np.random.seed(seed)
    return np.random.normal(loc=true_mean, scale=true_std, size=n_total)

def fit_normal_distribution(y_all, n_obs: int = 100, seed: int = 45):
    """Fit normal distribution to a subset of data."""
    true_mean = y_all.mean()
    true_std = y_all.std()
    
    # Sample subset
    np.random.seed(seed)
    idx = np.random.choice(len(y_all), n_obs, replace=False)
    y_sample = y_all[idx]
    
    # Fit parameters
    sample_mean = y_sample.mean()
    sample_std = y_sample.std(ddof=1)  # Sample std with Bessel's correction
    
    # Create distributions
    true_dist = norm(loc=true_mean, scale=true_std)
    fitted_dist = norm(loc=sample_mean, scale=sample_std)
    mean_dist = norm(loc=sample_mean, scale=true_std/np.sqrt(n_obs))  # Distribution of sample mean
    
    # Calculate confidence intervals
    ci_level = 0.9
    alpha = 1 - ci_level
    
    # CI for individual values (prediction interval)
    pred_ci = fitted_dist.ppf([alpha/2, 1-alpha/2])
    
    # CI for the mean
    mean_ci = mean_dist.ppf([alpha/2, 1-alpha/2])
    
    return {
        'true_mean': true_mean,
        'true_std': true_std,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'sample_data': y_sample,
        'true_dist': true_dist,
        'fitted_dist': fitted_dist,
        'mean_dist': mean_dist,
        'pred_ci': pred_ci,
        'mean_ci': mean_ci,
        'n_obs': n_obs
    }

def create_distribution_plot(fit_results):
    """Create plotly figure showing distribution fitting."""
    
    # Create x-axis for plotting distributions
    y_min = min(fit_results['sample_data'].min(), fit_results['pred_ci'][0])
    y_max = max(fit_results['sample_data'].max(), fit_results['pred_ci'][1])
    y_range = y_max - y_min
    x_range = np.linspace(y_min - 0.2*y_range, y_max + 0.2*y_range, 1000)
    
    # Calculate PDFs
    true_pdf = fit_results['true_dist'].pdf(x_range)
    fitted_pdf = fit_results['fitted_dist'].pdf(x_range)
    
    fig = go.Figure()
    
    # True distribution
    fig.add_trace(go.Scatter(
        x=x_range, y=true_pdf,
        mode='lines',
        name='True Distribution',
        line=dict(color='blue', width=2)
    ))
    
    # True mean line
    fig.add_vline(
        x=fit_results['true_mean'],
        line=dict(color='blue', dash='dash', width=2),
        annotation_text="True μ"
    )
    
    # Fitted distribution
    fig.add_trace(go.Scatter(
        x=x_range, y=fitted_pdf,
        mode='lines',
        name='Fitted Distribution',
        line=dict(color='red', width=2),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    # Sample data points as vertical lines at bottom
    max_pdf = max(true_pdf.max(), fitted_pdf.max())
    for obs in fit_results['sample_data']:
        fig.add_shape(
            type="line",
            x0=obs, x1=obs,
            y0=0, y1=0.05*max_pdf,
            line=dict(color="black", width=1)
        )
    
    # Add one sample point to legend
    fig.add_trace(go.Scatter(
        x=[fit_results['sample_data'][0]], y=[0.025*max_pdf],
        mode='markers',
        marker=dict(color='black', size=2, symbol='line-ns-open'),
        name=f'Sample Data (n={fit_results["n_obs"]})',
        showlegend=True
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=[fit_results['sample_mean'], fit_results['sample_mean']],
        y=[0.6*max_pdf, 0.6*max_pdf],
        mode='markers',
        marker=dict(color='red', size=8),
        error_x=dict(
            type='data',
            array=[fit_results['pred_ci'][1] - fit_results['sample_mean']],
            arrayminus=[fit_results['sample_mean'] - fit_results['pred_ci'][0]]
        ),
        name='90% CI of μ'
    ))
    
    fig.update_layout(
        title=f'Normal Distribution Fitting (n={fit_results["n_obs"]} samples)',
        xaxis_title='Y',
        yaxis_title='Probability Density',
        hovermode='x unified',
        legend=dict(x=0.7, y=0.95),
        height=500
    )
    
    return fig

def create_progression_plot(sample_sizes, results_list):
    """Create plot showing how estimates improve with sample size."""
    
    if not results_list:
        return go.Figure()
    
    # Extract data
    n_obs = np.array(sample_sizes)
    sample_means = np.array([r['sample_mean'] for r in results_list])
    mean_cis = np.array([r['mean_ci'] for r in results_list])
    pred_cis = np.array([r['pred_ci'] for r in results_list])
    
    true_mean = results_list[0]['true_mean']
    true_std = results_list[0]['true_std']
    true_dist = norm(loc=true_mean, scale=true_std)
    true_quantiles = true_dist.ppf([0.05, 0.95])
    
    fig = go.Figure()
    
    # True mean
    fig.add_hline(
        y=true_mean,
        line=dict(color='blue', width=2),
        annotation_text="True μ"
    )
    
    # True quantiles
    fig.add_hline(
        y=true_quantiles[0],
        line=dict(color='blue', dash='dash', width=1),
        annotation_text="True 90% quantile"
    )
    fig.add_hline(
        y=true_quantiles[1],
        line=dict(color='blue', dash='dash', width=1)
    )
    
    # Fitted means with CI
    fig.add_trace(go.Scatter(
        x=n_obs, y=sample_means,
        mode='lines+markers',
        name='Fitted μ',
        line=dict(color='red'),
        error_y=dict(
            type='data',
            array=mean_cis[:, 1] - sample_means,
            arrayminus=sample_means - mean_cis[:, 0],
            visible=True
        )
    ))
    
    # Prediction intervals
    fig.add_trace(go.Scatter(
        x=n_obs, y=sample_means,
        mode='lines',
        name='90% Quantiles',
        line=dict(color='red', dash='dash'),
        error_y=dict(
            type='data',
            array=pred_cis[:, 1] - sample_means,
            arrayminus=sample_means - pred_cis[:, 0],
            visible=True,
            thickness=1
        ),
        showlegend=True
    ))
    
    fig.update_layout(
        title='Convergence of Normal Distribution Fitting',
        xaxis_title='Number of Observations',
        yaxis_title='Y',
        xaxis_type='log',
        hovermode='x unified',
        legend=dict(x=0.7, y=0.95),
        height=500
    )
    
    return fig

# Layout
layout = html.Div([
    html.H1("Normal Distribution Fitting Analysis",
            style={"color": "#007bff", "marginBottom": "2rem", "textAlign": "center"}),
    
    html.P([
        "Explore how normal distribution parameter estimation improves with sample size. ",
        "This interactive tool demonstrates the Central Limit Theorem and confidence intervals."
    ], style={"textAlign": "center", "marginBottom": "2rem", "color": "#6c757d"}),
    
    # Controls
    html.Div([
        html.Div([
            html.H4("True Distribution Parameters"),
            html.Label("True Mean (μ):"),
            dcc.Slider(
                id='true-mean-slider',
                min=0, max=20, step=0.5, value=10,
                marks={i: str(i) for i in range(0, 21, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.Label("True Standard Deviation (σ):"),
            dcc.Slider(
                id='true-std-slider',
                min=0.5, max=5, step=0.1, value=2,
                marks={i: str(i) for i in [0.5, 1, 2, 3, 4, 5]},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '1rem'}),
        
        html.Div([
            html.H4("Sample Parameters"),
            html.Label("Sample Size (n):"),
            dcc.Slider(
                id='sample-size-slider',
                min=5, max=1000, step=1, value=100,
                marks={i: str(i) for i in [5, 10, 20, 50, 100, 200, 500, 1000]},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.Label("Random Seed:"),
            dcc.Input(
                id='random-seed-input',
                type='number',
                value=42,
                min=1, max=9999,
                style={'width': '100px', 'marginLeft': '10px'}
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '1rem'})
    ], style={'marginBottom': '2rem', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # Buttons
    html.Div([
        html.Button('Fit Single Distribution', id='fit-single-btn', n_clicks=0,
                   style={'margin': '0.5rem', 'padding': '0.5rem 1rem', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
        html.Button('Run Progression Analysis', id='fit-progression-btn', n_clicks=0,
                   style={'margin': '0.5rem', 'padding': '0.5rem 1rem', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
        html.Button('Clear Results', id='clear-btn', n_clicks=0,
                   style={'margin': '0.5rem', 'padding': '0.5rem 1rem', 'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none', 'borderRadius': '4px'})
    ], style={'textAlign': 'center', 'marginBottom': '2rem'}),
    
    # Results area
    html.Div(id='results-area'),
    
    # Store for data
    dcc.Store(id='progression-data', data=[])
    
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '2rem'})

@callback(
    Output('results-area', 'children'),
    Output('progression-data', 'data'),
    [Input('fit-single-btn', 'n_clicks'),
     Input('fit-progression-btn', 'n_clicks'),
     Input('clear-btn', 'n_clicks')],
    [State('true-mean-slider', 'value'),
     State('true-std-slider', 'value'),
     State('sample-size-slider', 'value'),
     State('random-seed-input', 'value'),
     State('progression-data', 'data')]
)
def update_results(fit_single_clicks, fit_progression_clicks, clear_clicks,
                  true_mean, true_std, sample_size, seed, stored_data):
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div(), []
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'clear-btn':
        return html.Div(), []
    
    # Generate data
    total_data = generate_normal_data(n_total=100000, true_mean=true_mean, true_std=true_std, seed=seed)
    
    if button_id == 'fit-single-btn':
        # Single fit
        result = fit_normal_distribution(total_data, n_obs=sample_size, seed=seed)
        fig = create_distribution_plot(result)
        
        # Summary statistics
        summary = html.Div([
            html.H4("Fitting Results"),
            html.P(f"True μ = {result['true_mean']:.3f}, True σ = {result['true_std']:.3f}"),
            html.P(f"Sample μ = {result['sample_mean']:.3f}, Sample σ = {result['sample_std']:.3f}"),
            html.P(f"90% CI for μ: [{result['mean_ci'][0]:.3f}, {result['mean_ci'][1]:.3f}]"),
            html.P(f"Sample size: {result['n_obs']}")
        ], style={'padding': '1rem', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '1rem 0'})
        
        return html.Div([
            summary,
            dcc.Graph(figure=fig)
        ]), stored_data
        
    elif button_id == 'fit-progression-btn':
        # Progression analysis
        sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        results = []
        
        for n in sample_sizes:
            if n <= len(total_data):
                result = fit_normal_distribution(total_data, n_obs=n, seed=seed)
                results.append(result)
        
        valid_sizes = [r['n_obs'] for r in results]
        
        if results:
            prog_fig = create_progression_plot(valid_sizes, results)
            
            # Create individual plots for key sample sizes
            key_sizes = [10, 100, 1000]
            individual_figs = []
            for n in key_sizes:
                if n in valid_sizes:
                    idx = valid_sizes.index(n)
                    fig = create_distribution_plot(results[idx])
                    individual_figs.append(dcc.Graph(figure=fig))
            
            # Store only JSON-serializable data (exclude scipy distribution objects)
            serializable_results = []
            for r in results:
                serializable_results.append({
                    'true_mean': r['true_mean'],
                    'true_std': r['true_std'],
                    'sample_mean': r['sample_mean'],
                    'sample_std': r['sample_std'],
                    'pred_ci': r['pred_ci'].tolist(),
                    'mean_ci': r['mean_ci'].tolist(),
                    'n_obs': r['n_obs']
                })
            
            return html.Div([
                html.H4("Progression Analysis Results"),
                dcc.Graph(figure=prog_fig),
                html.H4("Individual Fits for Key Sample Sizes"),
                html.Div(individual_figs, style={'display': 'flex', 'flexDirection': 'column'})
            ]), serializable_results
        
    return html.Div(), stored_data