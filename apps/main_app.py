"""
Central Application for Statistical and Probabilistic analysis in Geotechnics
=======================================================
This is the main entry point for all Statistical and Probabilistic analysis applications.
Navigate between different analysis tools using the sidebar.
"""

import dash
from dash import Dash, dcc, html, Input, Output, callback

# Initialize the main Dash app with pages support FIRST
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Statistical and Probabilistic analysis visualization"
)

# Import all pages to register them AFTER app instantiation
from pages import home, rf_visualization, foundation_analysis, pile_analysis, normal_distribution_fitting, settlement_analysis

# Define the sidebar navigation
sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6"
}

content_style = {
    "marginLeft": "18rem",
    "padding": "2rem 1rem"
}

sidebar = html.Div([
    html.H2("Statistical and Probabilistic analysis", style={"color": "#007bff", "marginBottom": "2rem"}),
    
    html.Hr(),
    
    dcc.Link("Home", href="/",
             style={
                 "display": "block", "padding": "0.75rem", "marginBottom": "0.5rem",
                 "textDecoration": "none", "color": "#495057", "borderRadius": "0.25rem",
                 "backgroundColor": "transparent"
             },
             className="nav-link"),
    
    dcc.Link([
        html.Div("Random Field Visualization", style={"fontWeight": "500"}),
        html.Small("Generate and visualize random fields", style={"color": "#6c757d"})
    ], href="/rf-visualization",
       style={
           "display": "block", "padding": "0.75rem", "marginBottom": "0.5rem",
           "textDecoration": "none", "color": "#495057", "borderRadius": "0.25rem",
           "backgroundColor": "transparent"
       }),
    
    dcc.Link([
        html.Div("Foundation Analysis", style={"fontWeight": "500"}),
        html.Small("Analyze foundation stability", style={"color": "#6c757d"})
    ], href="/foundation-analysis",
       style={
           "display": "block", "padding": "0.75rem", "marginBottom": "0.5rem",
           "textDecoration": "none", "color": "#495057", "borderRadius": "0.25rem",
           "backgroundColor": "transparent"
       }),
    
    dcc.Link([
        html.Div("Pile Analysis", style={"fontWeight": "500"}),
        html.Small("Analyze pile foundation systems", style={"color": "#6c757d"})
    ], href="/pile-analysis",
       style={
           "display": "block", "padding": "0.75rem", "marginBottom": "0.5rem",
           "textDecoration": "none", "color": "#495057", "borderRadius": "0.25rem",
           "backgroundColor": "transparent"
       }),
    
    dcc.Link([
        html.Div("Normal Distribution Fitting", style={"fontWeight": "500"}),
        html.Small("Fit normal distributions with varying sample sizes", style={"color": "#6c757d"})
    ], href="/normal-fitting",
       style={
           "display": "block", "padding": "0.75rem", "marginBottom": "0.5rem",
           "textDecoration": "none", "color": "#495057", "borderRadius": "0.25rem",
           "backgroundColor": "transparent"
       }),
    
    dcc.Link([
        html.Div("Settlement Analysis", style={"fontWeight": "500"}),
        html.Small("Interactive Bayesian settlement prediction analysis", style={"color": "#6c757d"})
    ], href="/settlement-analysis",
       style={
           "display": "block", "padding": "0.75rem", "marginBottom": "0.5rem",
           "textDecoration": "none", "color": "#495057", "borderRadius": "0.25rem",
           "backgroundColor": "transparent"
       }),
       
], style=sidebar_style)

# Main layout with sidebar and page content
app.layout = html.Div([
    sidebar,
    html.Div([
        dash.page_container
    ], style=content_style)
])


if __name__ == "__main__":

    app.run(debug=False, port=8050)

