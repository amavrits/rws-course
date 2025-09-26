"""
Foundation Analysis Page
========================
Analyze shallow foundation stability using random field modeling.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, callback
import numpy as np
import io, base64
import sys
import os
# Add the main directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from main.run_rf_foundation import foundation_analysis
    import matplotlib.pyplot as plt
except ImportError:
    # Handle missing imports gracefully
    def foundation_analysis(*args, **kwargs):
        raise ImportError("run_rf_foundation module not found")

# Register page only if dash pages is enabled
try:
    dash.register_page(__name__, path='/foundation-analysis', title='Foundation Analysis')
except:
    pass

# Controls component
controls = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "10px",
        "maxWidth": "300px",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "8px",
        "border": "1px solid #dee2e6"
    },
    children=[
        html.H4("Analysis Parameters", style={"marginBottom": "15px", "color": "#495057"}),
        
        html.Div([
            html.Label("Horizontal θ [m]", style={"marginBottom": "4px", "fontWeight": "500"}),
            dcc.Input(
                id="found-theta-x",
                type="number",
                value=100.0,
                min=1e-3,
                step="any",
                debounce=True,
                style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"}),
        
        html.Div([
            html.Label("Vertical θ [m]", style={"marginBottom": "4px", "fontWeight": "500"}),
            dcc.Input(
                id="found-theta-y",
                type="number",
                value=1.0,
                min=1e-3,
                step="any",
                debounce=True,
                style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"}),
        
        html.Div([
            html.Label("Foundation Width [m]", style={"marginBottom": "8px", "fontWeight": "500"}),
            dcc.Slider(
                id="found-foundation-width",
                min=1,
                max=50,
                step=1,
                value=10,
                marks={v: str(v) for v in [1, 5, 10, 20, 30, 40, 50]},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="mouseup",
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "20px"}),
        
        html.Div([
            html.Label("Random Seed", style={"marginBottom": "4px", "fontWeight": "500"}),
            dcc.Input(
                id="found-random-seed",
                type="number",
                value=42,
                step=1,
                debounce=True,
                style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"}),
        
        html.Button(
            "Run Foundation Analysis", 
            id="found-go", 
            n_clicks=0, 
            style={
                "height": "40px", 
                "backgroundColor": "#28a745", 
                "color": "white", 
                "border": "none", 
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontWeight": "500"
            }
        ),
    ],
)

# Page layout
layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Foundation Analysis", style={"marginBottom": "10px", "color": "#343a40"}),
        html.P([
            "Analyze shallow foundation stability considering spatial variability of soil strength. ",
            "This tool uses random field modeling to assess foundation safety factors and provides ",
            "probabilistic insights into foundation performance."
        ], style={"marginBottom": "30px", "color": "#6c757d", "fontSize": "16px"}),
        
        html.Div([
            controls,
            html.Div([
                html.Div(
                    id="found-status", 
                    style={
                        "marginBottom": "15px", 
                        "color": "#666", 
                        "fontFamily": "monospace",
                        "fontSize": "14px",
                        "padding": "10px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "4px",
                        "minHeight": "20px"
                    }
                ),
                dcc.Loading(
                    id="found-img-loading",
                    type="default",
                    children=html.Img(
                        id="found-img",
                        style={
                            "display": "block", 
                            "maxWidth": "100%", 
                            "width": "100%",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "8px"
                        }
                    )
                ),
            ], style={"flex": "1", "marginLeft": "30px"})
        ], style={"display": "flex", "alignItems": "flex-start"}),
        
        html.Div([
            html.H3("Analysis Information", style={"marginTop": "40px", "marginBottom": "15px"}),
            html.Ul([
                html.Li("Foundation load: 400 kN/m²"),
                html.Li("Soil mean strength: 20 kPa"),
                html.Li("Soil strength standard deviation: 4 kPa"),
                html.Li("Grid resolution: 100×50 points"),
                html.Li("Analysis includes safety factor calculation and risk assessment")
            ], style={"color": "#6c757d"})
        ])
    ],
)

@callback(
    Output("found-img", "src"),
    Output("found-status", "children"),
    Input("found-go", "n_clicks"),
    State("found-foundation-width", "value"),
    State("found-theta-x", "value"),
    State("found-theta-y", "value"),
    State("found-random-seed", "value"),
    prevent_initial_call=True,
)
def compute_and_show_foundation(n_clicks, width, tx, ty, random_seed):
    if tx is None or ty is None:
        return no_update, "⚠️ Enter valid horizontal and vertical θ's (>0.001)."
    if tx <= 0 or ty <= 0:
        return no_update, "⚠️ The horizontal and vertical θ's must be positive."

    try:
        fos, fig = foundation_analysis(
            foundation_width=width,
            theta_x=tx,
            theta_y=ty,
            mean=20.,
            std=4.,
            n_x=100,
            n_y=50,
            foundation_load=400.,
            path=None,
            return_fig=True,
            random_seed=random_seed
        )

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        return no_update, f"❌ Error: {type(e).__name__}: {e}"

    return src, f"✅ Analysis complete: Foundation width={int(width)}m, θx={tx:.2f}m, θy={ty:.2f}m, FoS≈{fos:.2f}"