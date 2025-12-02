"""
Central Application for Statistical and Probabilistic Analysis in Geotechnics
=============================================================================

This is the main entry point for all Statistical and Probabilistic analysis
applications. Navigate between different analysis tools using the sidebar.
"""

import dash
from dash import Dash, dcc, html

# Initialize the main Dash app with pages support FIRST
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Statistical and Probabilistic Analysis Visualization",
)

# Expose the Flask server for deployment (gunicorn, etc.)
server = app.server

print("Pages:", dash.page_registry.keys())


# --- Layout styles ---

SIDEBAR_WIDTH = "18rem"

sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": SIDEBAR_WIDTH,
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "overflowY": "auto",
}

content_style = {
    "marginLeft": SIDEBAR_WIDTH,
    "padding": "1rem 0.5rem",
}

link_base_style = {
    "display": "block",
    "padding": "0.75rem",
    "marginBottom": "0.5rem",
    "textDecoration": "none",
    "color": "#495057",
    "borderRadius": "0.25rem",
    "backgroundColor": "transparent",
}

sidebar = html.Div(
    [
        html.H2(
            "Statistical and Probabilistic analysis",
            style={"color": "#007bff", "marginBottom": "2rem"},
        ),
        html.Hr(),
        dcc.Link(
            "Home",
            href="/",
            style=link_base_style,
            className="nav-link",
        ),
        dcc.Link(
            [
                html.Div("Bayesian coin experiment visualization", style={"fontWeight": "500"}),
                html.Small(
                    "Generate, solve and visualize the coin experiment using Bayesian statistics",
                    style={"color": "#6c757d"},
                ),
            ],
            href="/bayesian-coin-analysis",
            style=link_base_style,
        ),
        dcc.Link(
            [
                html.Div("Random Field Visualization", style={"fontWeight": "500"}),
                html.Small("Generate and visualize random fields", style={"color": "#6c757d"}),
            ],
            href="/rf-visualization",
            style=link_base_style,
        ),
        dcc.Link(
            [
                html.Div("Foundation Analysis", style={"fontWeight": "500"}),
                html.Small("Analyze foundation stability", style={"color": "#6c757d"}),
            ],
            href="/foundation-analysis",
            style=link_base_style,
        ),
        dcc.Link(
            [
                html.Div("Pile Analysis", style={"fontWeight": "500"}),
                html.Small("Analyze pile foundation systems", style={"color": "#6c757d"}),
            ],
            href="/pile-analysis",
            style=link_base_style,
        ),
        dcc.Link(
            [
                html.Div("Normal Distribution Fitting", style={"fontWeight": "500"}),
                html.Small("Fit normal distributions with varying sample sizes", style={"color": "#6c757d"}),
            ],
            href="/normal-fitting",
            style=link_base_style,
        ),
        dcc.Link(
            [
                html.Div("Settlement Analysis", style={"fontWeight": "500"}),
                html.Small(
                    "Interactive Bayesian settlement prediction analysis",
                    style={"color": "#6c757d"},
                ),
            ],
            href="/settlement-analysis",
            style=link_base_style,
        ),
    ],
    style=sidebar_style,
)

# Main layout with sidebar and page content
app.layout = html.Div(
    [
        sidebar,
        html.Div(
            dash.page_container,
            style=content_style,
        ),
    ]
)


if __name__ == "__main__":

    # host="0.0.0.0" is handy if you want to reach it from another machine / Docker
    app.run(debug=False, host="0.0.0.0", port=8050)

