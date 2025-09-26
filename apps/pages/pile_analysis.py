"""
Pile Analysis Page
==================
Analyze pile foundation systems using random field modeling.
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
    from main.run_rf_piles import pile_analysis
    import matplotlib.pyplot as plt
except ImportError:
    # Handle missing imports gracefully
    def pile_analysis(*args, **kwargs):
        raise ImportError("run_rf_piles module not found")

# Register page only if dash pages is enabled
try:
    dash.register_page(__name__, path='/pile-analysis', title='Pile Analysis')
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
                id="pile-theta-x",
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
                id="pile-theta-y",
                type="number",
                value=1.0,
                min=1e-3,
                step="any",
                debounce=True,
                style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"}),
        
        html.Div([
            html.Label("Number of Piles", style={"marginBottom": "8px", "fontWeight": "500"}),
            dcc.Slider(
                id="pile-n-piles",
                min=1,
                max=40,
                step=1,
                value=10,
                marks={v: str(v) for v in [1, 5, 10, 15, 20, 30, 40]},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="mouseup",
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "20px"}),
        
        html.Div([
            html.Label("Random Seed", style={"marginBottom": "4px", "fontWeight": "500"}),
            dcc.Input(
                id="pile-random-seed",
                type="number",
                value=42,
                step=1,
                debounce=True,
                style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
            ),
        ], style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"}),
        
        html.Button(
            "Run Pile Analysis", 
            id="pile-go", 
            n_clicks=0, 
            style={
                "height": "40px", 
                "backgroundColor": "#ffc107", 
                "color": "#212529", 
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
        html.H1("Pile Foundation Analysis", style={"marginBottom": "10px", "color": "#343a40"}),
        html.P([
            "Evaluate pile foundation systems with varying numbers of piles. ",
            "This analysis considers load distribution and spatial correlation effects ",
            "on foundation performance using random field modeling techniques."
        ], style={"marginBottom": "30px", "color": "#6c757d", "fontSize": "16px"}),
        
        html.Div([
            controls,
            html.Div([
                html.Div(
                    id="pile-status", 
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
                    id="pile-img-loading",
                    type="default",
                    children=html.Img(
                        id="pile-img",
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
                html.Li("Load per pile: 10 kN"),
                html.Li("Pile diameter: 1.0 m"),
                html.Li("Soil mean strength: 20 kPa"),
                html.Li("Soil strength standard deviation: 4 kPa"),
                html.Li("Grid resolution: 100×50 points"),
                html.Li("Analysis includes pile group effects and load distribution")
            ], style={"color": "#6c757d"})
        ])
    ],
)

@callback(
    Output("pile-img", "src"),
    Output("pile-status", "children"),
    Input("pile-go", "n_clicks"),
    State("pile-n-piles", "value"),
    State("pile-theta-x", "value"),
    State("pile-theta-y", "value"),
    State("pile-random-seed", "value"),
    prevent_initial_call=True,
)
def compute_and_show_pile(n_clicks, n_piles, tx, ty, random_seed):
    if tx is None or ty is None:
        return no_update, "⚠️ Enter valid horizontal and vertical θ's (>0.001)."
    if tx <= 0 or ty <= 0:
        return no_update, "⚠️ The horizontal and vertical θ's must be positive."

    try:
        fos, fig = pile_analysis(
            n_piles=n_piles,
            theta_x=tx,
            theta_y=ty,
            mean=20.,
            std=4.,
            n_x=100,
            n_y=50,
            load_per_pile=10.,
            pile_diameter=1.,
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

    return src, f"✅ Analysis complete: {n_piles} piles, θx={tx:.2f}m, θy={ty:.2f}m, FoS≈{fos:.2f}"