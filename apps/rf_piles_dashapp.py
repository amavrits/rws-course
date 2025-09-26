from dash import Dash, dcc, html, Input, Output, State, no_update
from main.run_rf_piles import *
from pathlib import Path
import threading
import numpy as np
import io, base64


app = Dash(__name__)
app.title = "Foundation analysis using Random Field modeling"


controls = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "10px",
        "maxWidth": "280px",
    },
    children=[
        html.Div(
            [
                html.Label("Horizontal θ [m]", style={"marginBottom": "4px"}),
                dcc.Input(
                    id="theta-x",
                    type="number",
                    value=100.0,
                    min=1e-3,
                    step="any",
                    debounce=True,
                    style={"width": "100%"},
                ),
            ],
            style={"display": "flex", "flexDirection": "column"},
        ),
        html.Div(
            [
                html.Label("Vertical θ [m]", style={"marginBottom": "4px"}),
                dcc.Input(
                    id="theta-y",
                    type="number",
                    value=1.0,
                    min=1e-3,
                    step="any",
                    debounce=True,
                    style={"width": "100%"},
                ),
            ],
            style={"display": "flex", "flexDirection": "column"},
        ),
        html.Div(
            [
                html.Label("Number of piles", style={"marginBottom": "4px"}),
                dcc.Slider(
                    id="n-piles",
                    min=1,
                    max=40,
                    step=None,
                    value=10,
                    marks={v: str(v) for v in [1, 5, 10, 20, 40]},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="mouseup",
                ),
            ],
            style={"display": "flex", "flexDirection": "column"},
        ),
        html.Div(
            [
                html.Label("Random seed", style={"marginBottom": "4px"}),
                dcc.Input(
                    id="random-seed",
                    type="number",
                    value=42,
                    step="int",
                    debounce=True,
                    style={"width": "100%"},
                ),
            ],
            style={"display": "flex", "flexDirection": "column"},
        ),
        html.Button("Compute", id="go", n_clicks=0, style={"height": "38px"}),
    ],
)

# Then in app.layout, use:
app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "24px"},
    children=[
        html.H3("Foundation analysis using Random Field modeling"),
        controls,
        html.Div(id="status", style={"marginTop": "8px", "color": "#666", "fontFamily": "monospace"}),
        dcc.Loading(
            id="img-loading",
            type="default",
            children=html.Img(
                id="rf-img",
                style={"display": "block", "maxWidth": "900px", "width": "100%", "marginTop": 12}
            )
        ),
    ],
)


@app.callback(
    Output("rf-img", "src"),
    Output("status", "children"),
    Input("go", "n_clicks"),
    State("n-piles", "value"),
    State("theta-x", "value"),
    State("theta-y", "value"),
    State("random-seed", "value"),
    prevent_initial_call=True,
)
def compute_and_show(n_clicks, n_piles, tx, ty, random_seed):

    if tx is None or ty is None:
        return no_update, "Enter valid horizontal and vertical θ's (>0.001)."
    if tx <= 0 or ty <= 0:
        return no_update, "The horizontal and vertical θ's must be positive."

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
        fig.savefig(buf, format="png")
        plt.close(fig)
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:

        return no_update, f"Error: {type(e).__name__}: {e}"

    return src, f"Rendered with horizontal θ={tx:.2f}, vertical θ={ty:.2f}, number of piles={n_piles}."


if __name__ == "__main__":

    app.run(debug=False)

