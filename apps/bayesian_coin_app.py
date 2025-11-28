from dash import Dash, dcc, html, Input, Output, State, no_update
from main.run_bayesian_coin_experiment import *
import numpy as np
import io, base64

N_MIN = 1
N_MAX = 10_000
GAMMA = 2.0   # >1 => more density at HIGH n


def slider_to_n(x: float) -> int:
    """
    x in [0, 1] -> n in [N_MIN, N_MAX]
    GAMMA > 1 => slower growth at low x, faster at high x
    """
    x = float(x)
    x = min(max(x, 0.0), 1.0)  # clamp

    t = x ** GAMMA
    n = N_MIN * (N_MAX / N_MIN) ** t
    return int(round(n))


def n_to_slider(n: float) -> float:
    """
    Inverse of slider_to_n for marks:
    n in [N_MIN, N_MAX] -> x in [0, 1]
    """
    n = float(n)
    n = min(max(n, N_MIN), N_MAX)  # clamp

    # t such that n = N_MIN * (N_MAX/N_MIN)**t
    t = np.log(n / N_MIN) / np.log(N_MAX / N_MIN)

    # x such that t = x**GAMMA -> x = t**(1/GAMMA)
    x = t ** (1.0 / GAMMA)
    return float(x)


desired_ns = [1, 10, 100, 1_000, 10_000]
marks = {
    n_to_slider(n): f"{n:,}"
    for n in desired_ns
}


app = Dash(__name__)
app.title = "Bayesian coin experiment visualization"

controls = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "30px",
        "width": "800px",
        "margin": "0 auto",
    },
    children=[
        html.Div(
            [
                html.Label("Number of observations (n)"),
                dcc.Slider(
                    id="n_slider",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    value=n_to_slider(100),  # default around n ≈ 100
                    marks=marks,
                    tooltip={"always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                html.Label("True coin probability (p)"),
                dcc.Slider(
                    id="true_p",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    marks={0: "0", 0.5: "0.5", 1: "1"},
                    tooltip={"always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Prior parameter a"),
                dcc.Slider(
                    id="prior_a",
                    min=0.1,
                    max=10,
                    step=0.1,
                    value=2,
                    marks={0: "0", 5: "5", 10: "10"},
                    tooltip={"always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Prior parameter b"),
                dcc.Slider(
                    id="prior_b",
                    min=0.1,
                    max=10,
                    step=0.1,
                    value=2,
                    marks={0: "0", 5: "5", 10: "10"},
                    tooltip={"always_visible": True},
                ),
            ]
        ),
    ]
)

app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "24px"},
    children=[
        html.H3("Bayesian coin experiment"),
        controls,
        html.Div(
            id="status",
            style={"marginTop": "8px", "color": "#666", "fontFamily": "monospace"},
        ),
        dcc.Loading(
            id="img-loading",
            type="default",
            children=html.Img(
                id="rf-img",
                style={
                    "display": "block",
                    "margin": "20px auto",
                    "maxWidth": "900px",
                    "width": "100%",
                }
            ),
        ),
    ],
)


@app.callback(
    Output("rf-img", "src"),
    Output("status", "children"),
    Input("n_slider", "value"),        # 🔁 trigger on any change
    Input("true_p", "value"),
    Input("prior_a", "value"),
    Input("prior_b", "value"),
)
def compute_and_show(n_slider, true_p, prior_a, prior_b):

    try:

        n = slider_to_n(n_slider)

        true_dist = bernoulli(p=true_p)
        np.random.seed(42)
        sample = true_dist.rvs(n)

        prior_params = (prior_a, prior_b)

        posterior_params = inference(sample, prior_params)

        fig = plot_cis(
            prior_params=prior_params,
            posterior_params=posterior_params,
            n=n,
            true_p=true_p,
            return_fig=True
        )

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:

        return no_update, f"Error: {type(e).__name__}: {e}"

    return src, ""


if __name__ == "__main__":

    app.run(debug=False)

