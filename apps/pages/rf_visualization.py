"""
Random Field Visualization Page
===============================
Interactive visualization of 2D random fields with customizable correlation parameters.
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
    from main.generate_rf import make_rf
    import matplotlib.pyplot as plt
except ImportError:
    # Handle missing imports gracefully
    def make_rf(*args, **kwargs):
        raise ImportError("generate_rf module not found")

# Register page only if dash pages is enabled
try:
    dash.register_page(__name__, path='/rf-visualization', title='Random Field Visualization')
except:
    pass

# Grid setup
x_grid = np.linspace(0, 100, 100)
y_grid = np.linspace(0, 20, 50)
coords = np.meshgrid(x_grid, y_grid)
coords = np.c_[[m.flatten() for m in coords]].T

# Controls component
controls = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "10px",
        "maxWidth": "280px",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "8px",
        "border": "1px solid #dee2e6"
    },
    children=[
        html.H4("Parameters", style={"marginBottom": "15px", "color": "#495057"}),
        html.Div(
            [
                html.Label("Horizontal θ [m]", style={"marginBottom": "4px", "fontWeight": "500"}),
                dcc.Input(
                    id="rf-viz-theta-x",
                    type="number",
                    value=100.0,
                    min=1e-3,
                    step="any",
                    debounce=True,
                    style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"},
        ),
        html.Div(
            [
                html.Label("Vertical θ [m]", style={"marginBottom": "4px", "fontWeight": "500"}),
                dcc.Input(
                    id="rf-viz-theta-y",
                    type="number",
                    value=1.0,
                    min=1e-3,
                    step="any",
                    debounce=True,
                    style={"width": "100%", "padding": "8px", "borderRadius": "4px"},
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "marginBottom": "15px"},
        ),
        html.Button(
            "Generate Random Field", 
            id="rf-viz-go", 
            n_clicks=0, 
            style={
                "height": "40px", 
                "backgroundColor": "#007bff", 
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
        html.H1("Random Field Visualization", style={"marginBottom": "10px", "color": "#343a40"}),
        html.P(
            "Generate and visualize 2D random fields with customizable correlation parameters. "
            "Adjust the horizontal and vertical correlation lengths (θ) to see how they affect "
            "the spatial structure of the random field.",
            style={"marginBottom": "30px", "color": "#6c757d", "fontSize": "16px"}
        ),
        
        html.Div([
            controls,
            html.Div([
                html.Div(
                    id="rf-viz-status", 
                    style={
                        "marginBottom": "15px", 
                        "color": "#666", 
                        "fontFamily": "monospace",
                        "fontSize": "14px",
                        "padding": "10px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "4px"
                    }
                ),
                dcc.Loading(
                    id="rf-viz-img-loading",
                    type="default",
                    children=html.Img(
                        id="rf-viz-img",
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
    ],
)

# Import the app instance from main_app
from dash import callback, Output, Input, State, no_update

@callback(
    Output("rf-viz-img", "src"),
    Output("rf-viz-status", "children"),
    Input("rf-viz-go", "n_clicks"),
    State("rf-viz-theta-x", "value"),
    State("rf-viz-theta-y", "value"),
    prevent_initial_call=True,
)
def compute_and_show_rf_viz(n_clicks, tx, ty):
    if tx is None or ty is None:
        return no_update, "⚠️ Enter valid horizontal and vertical θ's (>0.001)."
    if tx <= 0 or ty <= 0:
        return no_update, "⚠️ The horizontal and vertical θ's must be positive."

    try:
        fig = make_rf(
            coords=coords,
            mean=20.,
            std=4.,
            n_x=x_grid.size,
            n_y=y_grid.size,
            theta_x=tx,
            theta_y=ty,
            return_fig=True
        )

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        return no_update, f"❌ Error: {type(e).__name__}: {e}"

    return src, f"✅ Random field generated with horizontal θ={tx:.2f}m, vertical θ={ty:.2f}m"