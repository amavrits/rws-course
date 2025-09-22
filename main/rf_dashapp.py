from dash import Dash, dcc, html, Input, Output, State, no_update
from main.generate_rf import *
from pathlib import Path
import threading
import numpy as np
import io, base64


x_grid = np.linspace(0, 100, 100)
y_grid = np.linspace(0, 20, 50)
coords = np.meshgrid(x_grid, y_grid)
coords = np.c_[[m.flatten() for m in coords]].T


app = Dash(__name__)
app.title = "Random Field visualization"


app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "24px"},
    children=[
        html.H3("Random Field visualization"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0, 1fr))", "gap": "12px"},
            children=[
                html.Label("Horizontal θ [m]", style={"gridColumn": "span 1", "alignSelf": "end"}),
                dcc.Input(id="theta-x", type="number", value=100.0, min=1e-3, step="any", debounce=True),
                html.Label("Vertical θ [m]", style={"gridColumn": "span 1", "alignSelf": "end"}),
                dcc.Input(id="theta-y", type="number", value=1.0, min=1e-3, step="any", debounce=True),
                html.Button("Compute", id="go", n_clicks=0, style={"gridColumn": "span 6", "height": "38px"}),
            ],
        ),
        html.Div(id="status", style={"marginTop": "8px", "color": "#666", "fontFamily": "monospace"}),
        dcc.Loading(
            id="img-loading",
            type="default",
            children=html.Img(
                id="rf-img",
                style={"display": "block", "maxWidth": "900px", "width": "100%", "marginTop": 12}
                )
        )
    ],
)


@app.callback(
    Output("rf-img", "src"),
    Output("status", "children"),
    Input("go", "n_clicks"),
    State("theta-x", "value"),
    State("theta-y", "value"),
    prevent_initial_call=True,
)
def compute_and_show(n_clicks, tx, ty):

    if tx is None or ty is None:
        return no_update, "Enter valid horizontal and vertical θ's (>0.001)."
    if tx <= 0 or ty <= 0:
        return no_update, "The horizontal and vertical θ's must be positive."

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
        fig.savefig(buf, format="png")
        plt.close(fig)
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:

        return no_update, f"Error: {type(e).__name__}: {e}"

    return src, f"Rendered with horizontal θ={tx:.2f}, vertical θ={ty:.2f}."


if __name__ == "__main__":

    app.run(debug=False)

