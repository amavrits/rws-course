"""
Home Page - Random Field Analysis Hub
=====================================
Welcome page with overview of available analysis tools.
"""

import dash
from dash import html, dcc

# Register page only if dash pages is enabled
try:
    dash.register_page(__name__, path='/', title='Probabilistic and Statistical Analysis Hub - Home')
except:
    pass

# Card styling
card_style = {
    "border": "1px solid #dee2e6",
    "borderRadius": "8px",
    "padding": "1.5rem",
    "margin": "1rem",
    "backgroundColor": "white",
    "boxShadow": "0 0.125rem 0.25rem rgba(0,0,0,0.075)",
    "height": "280px",
    "display": "flex",
    "flexDirection": "column"
}

button_style = {
    "padding": "0.5rem 1rem",
    "border": "none",
    "borderRadius": "4px",
    "textDecoration": "none",
    "display": "inline-block",
    "textAlign": "center",
    "cursor": "pointer",
    "fontWeight": "500",
    "marginTop": "auto"
}

layout = html.Div([
    html.H1("Random Field Analysis Hub",
            style={"fontSize": "3rem", "color": "#007bff", "marginBottom": "1rem", "textAlign": "center"}),
    html.P([
        "Welcome to the Random Field Analysis platform. This application provides ",
        "interactive tools for geotechnical analysis using random field modeling."
    ], style={"fontSize": "1.25rem", "marginBottom": "2rem", "textAlign": "center", "color": "#6c757d"}),

    # Cards section
    html.Div([

        html.Div([
            html.H4("Random Field Visualization", style={"marginBottom": "1rem"}),
            html.P([
                "Generate a coin experiment, solve using Bayesian statistics (beta-bernoulli conjugate priors)"
                "and visualize.",
            ], style={"marginBottom": "1rem", "flex": "1"}),
            dcc.Link("Open Visualization Tool",
                     href="/bayesian-coin-analysis",
                     style={**button_style, "backgroundColor": "#007bff", "color": "white"})
        ], style=card_style),

        html.Div([
            html.H4("Random Field Visualization", style={"marginBottom": "1rem"}),
            html.P([
                "Generate and visualize 2D random fields with customizable ",
                "correlation parameters. Perfect for understanding spatial ",
                "variability in soil properties."
            ], style={"marginBottom": "1rem", "flex": "1"}),
            dcc.Link("Open Visualization Tool",
                    href="/rf-visualization",
                    style={**button_style, "backgroundColor": "#007bff", "color": "white"})
        ], style=card_style),
        
        html.Div([
            html.H4("Foundation Analysis", style={"marginBottom": "1rem"}),
            html.P([
                "Analyze shallow foundation stability considering spatial ",
                "variability of soil strength. Includes safety factor ",
                "calculations and risk assessment."
            ], style={"marginBottom": "1rem", "flex": "1"}),
            dcc.Link("Open Foundation Tool", 
                    href="/foundation-analysis", 
                    style={**button_style, "backgroundColor": "#28a745", "color": "white"})
        ], style=card_style),
        
        html.Div([
            html.H4("Pile Analysis", style={"marginBottom": "1rem"}),
            html.P([
                "Evaluate pile foundation systems with varying numbers of ",
                "piles. Considers load distribution and spatial correlation ",
                "effects on foundation performance."
            ], style={"marginBottom": "1rem", "flex": "1"}),
            dcc.Link("Open Pile Tool", 
                    href="/pile-analysis", 
                    style={**button_style, "backgroundColor": "#ffc107", "color": "#212529"})
        ], style=card_style),
        
        html.Div([
            html.H4("Normal Distribution Fitting", style={"marginBottom": "1rem"}),
            html.P([
                "Interactive exploration of normal distribution parameter ",
                "estimation with different sample sizes. Demonstrates the ",
                "Central Limit Theorem and confidence intervals."
            ], style={"marginBottom": "1rem", "flex": "1"}),
            dcc.Link("Open Normal Fitting Tool", 
                    href="/normal-fitting", 
                    style={**button_style, "backgroundColor": "#17a2b8", "color": "white"})
        ], style=card_style)
    ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "marginBottom": "3rem"}),
    
    html.Hr(style={"margin": "3rem 0"}),
    
    # Information section
    html.Div([
        html.Div([
            html.H3("Getting Started", style={"marginBottom": "1rem"}),
            html.Ol([
                html.Li("Select an analysis tool from the sidebar or cards above"),
                html.Li("Adjust the input parameters (correlation lengths, dimensions, etc.)"),
                html.Li("Click 'Compute' to run the analysis"),
                html.Li("Explore the visualizations and results")
            ], style={"marginBottom": "2rem"}),
            
            html.H3("Key Features", style={"marginBottom": "1rem"}),
            html.Ul([
                html.Li("Interactive parameter adjustment with real-time feedback"),
                html.Li("High-quality visualizations of random fields and analysis results"),
                html.Li("Probabilistic analysis for geotechnical design"),
                html.Li("Customizable random seeds for reproducible results")
            ])
        ], style={"flex": "2", "marginRight": "2rem"}),
        
        html.Div([
            html.Div([
                html.H4("Pro Tip", style={"color": "#0c5460", "marginBottom": "1rem"}),
                html.P([
                    "Start with the Random Field Visualization to understand ",
                    "how correlation parameters affect spatial patterns, then ",
                    "proceed to the engineering analysis tools."
                ])
            ], style={
                "backgroundColor": "#d1ecf1", 
                "border": "1px solid #bee5eb", 
                "borderRadius": "4px",
                "padding": "1rem"
            })
        ], style={"flex": "1"})
    ], style={"display": "flex", "marginTop": "2rem"})
], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "2rem"})