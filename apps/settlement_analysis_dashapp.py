"""
Settlement Analysis Visualization Dashboard
==========================================
This Dash app provides an interactive visualization of settlement analysis predictions.
Users can navigate through different time points to see how predictions evolve.
"""

from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from numpy.typing import NDArray


def load_predictions_data():
    """Load predictions data from the JSON file."""
    data_path = Path(__file__).parent.parent / "data/settlement_analysis/predictions.json"
    with open(data_path, "r") as f:
        predictions = json.load(f)
    return predictions


def plot_predictions(
        predictions: Dict[str, List[float]],
        true_cv: float,
        true_settlement: NDArray,
        target_settlement: float,
        path: Optional[Path] = None,
        return_fig: bool = False
) -> Optional[plt.Figure]:
    
    all_times = predictions["all_times"]
    obs_times = predictions["observation_times"]
    forecast_times = predictions["forecast_times"]
    settlement_obs = predictions["observations"]
    prior_mean = predictions["prior_mean"]
    prior_lower_quantile = predictions["prior_lower_quantile"]
    prior_upper_quantile = predictions["prior_upper_quantile"]
    posterior_mean = predictions["posterior_mean"]
    posterior_lower_quantile = predictions["posterior_lower_quantile"]
    posterior_upper_quantile = predictions["posterior_upper_quantile"]
    cv_grid = predictions["cv_grid"]
    cv_prior_pdf = predictions["cv_prior_pdf"]
    cv_posterior_pdf = predictions["cv_posterior_pdf"]
    prior_exceeds_target = [p*100 for p in predictions["prior_exceeds_target"]]
    posterior_exceeds_target = [p*100 for p in predictions["posterior_exceeds_target"]]

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={"width_ratios": [3, 1]})

    ax = axs[0]

    if len(obs_times) > 0:
        ax.axvline(max(obs_times), c="k", linestyle="--")
        ax.scatter(obs_times, settlement_obs, color="k", marker="x", label="Observations")

    ax.plot(forecast_times, prior_mean, c="b", linewidth=1.5, label="Prior mean prediction")
    ax.fill_between(
        x=forecast_times,
        y1=prior_lower_quantile,
        y2=prior_upper_quantile,
        color="b", alpha=0.3, label="Prior 90% CI"
    )
    ax.plot(forecast_times, prior_lower_quantile, c="b", linewidth=.5)
    ax.plot(forecast_times, prior_upper_quantile, c="b", linewidth=.5)

    ax.plot(forecast_times, posterior_mean, c="r", linewidth=1.5, label="Posterior mean prediction")
    ax.fill_between(
        x=forecast_times,
        y1=posterior_lower_quantile,
        y2=posterior_upper_quantile,
        color="r", alpha=0.3, label="Posterior 90% CI"
    )
    ax.plot(forecast_times, posterior_lower_quantile, c="r", linewidth=.5)
    ax.plot(forecast_times, posterior_upper_quantile, c="r", linewidth=.5)

    ax.plot(all_times, true_settlement, c="g", linewidth=1.5, label="True settlement model")
    ax.axhline(target_settlement, c="k", linewidth=1.5, label="Target settlement")

    ax2 = ax.twinx()
    ax2.plot(forecast_times, prior_exceeds_target, c="b", linestyle="--", label="P(S>Target) - Prior")
    ax2.plot(forecast_times, posterior_exceeds_target, c="r", linestyle="--", label="P(S>Target) - Posterior")

    ax.set_xlabel("Time [d]", fontsize=12)
    ax.set_ylabel("Settlement [m]", fontsize=12)
    ax.invert_yaxis()
    ax.grid()
    ax2.set_ylabel("Target exceedance probability [%]", fontsize=12)
    ax2.set_ylim(0, 100)

    # Combine legends from both y-axes and place in upper left corner
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Alternative 1: Figure-level legend above left subplot (current approach)
    fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.375, 0.95),
              ncol=4, fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Alternative 2: If above doesn"t work well, uncomment this for upper left corner:
    # ax.legend(all_handles, all_labels, loc="upper left", fontsize=9,
    #          frameon=True, fancybox=True, shadow=True, ncol=2)


    ax = axs[1]
    ax.plot(cv_grid, cv_prior_pdf, c="b", label="Prior PDF")
    ax.plot(cv_grid, cv_posterior_pdf, c="r", label="Posterior PDF")
    ax.axvline(true_cv, linewidth=2, c="g", label="True ${C}_{v}$")
    ax.set_xlabel("${C}_{v}$ [${m}^{2}$/d]", fontsize=12)
    ax.set_ylabel("Density [-]", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the figure-level legend

    if return_fig:
        return fig
    else:
        if path is not None:
            path = path / "plots"
            path.mkdir(exist_ok=True, parents=True)
            if len(obs_times) > 0:
                fig.savefig(path/f"settlement_prediction_time_{max(obs_times)}.png", bbox_inches="tight")
            else:
                fig.savefig(path/"settlement_prediction_time_0.png", bbox_inches="tight")
        return


def get_legend_parts(item):
    """Extract color, symbol, and text from legend item."""
    # Determine color based on item type
    if "Prior" in item and "P(S>" not in item:
        color = "blue"
    elif "Posterior" in item and "P(S>" not in item:
        color = "red"
    elif "True" in item:
        color = "green"
    elif "Target" in item:
        color = "black"
    elif "P(S>Target) Prior" in item:
        color = "blue"
    elif "P(S>Target) Posterior" in item:
        color = "red"
    elif "Observations" in item:
        color = "black"
    else:
        color = "black"

    # Extract symbol and text
    if "🞩" in item:
        symbol, text = item.split(" ", 1)
    elif item.startswith(("━", "█", "┅")):
        symbol = item[0]
        text = item[2:]
    else:
        symbol = "━"
        text = item

    return color, symbol, text


def create_plotly_figure(predictions_data, time_key, true_cv=2.3*1e-8 * (24 * 3_600)):
    """Create a clean Plotly figure from predictions data."""
    prediction = predictions_data[str(time_key)]

    all_times = prediction["all_times"]
    obs_times = prediction["observation_times"]
    forecast_times = prediction["forecast_times"]
    settlement_obs = prediction["observations"]
    prior_mean = prediction["prior_mean"]
    prior_lower_quantile = prediction["prior_lower_quantile"]
    prior_upper_quantile = prediction["prior_upper_quantile"]
    posterior_mean = prediction["posterior_mean"]
    posterior_lower_quantile = prediction["posterior_lower_quantile"]
    posterior_upper_quantile = prediction["posterior_upper_quantile"]
    cv_grid = prediction["cv_grid"]
    cv_prior_pdf = prediction["cv_prior_pdf"]
    cv_posterior_pdf = prediction["cv_posterior_pdf"]

    # Get target exceedance probabilities (convert to percentages)
    try:
        prior_exceeds_target = [p*100 for p in prediction["prior_exceeds_target"]]
        posterior_exceeds_target = [p*100 for p in prediction["posterior_exceeds_target"]]
        has_target_data = True
    except KeyError:
        prior_exceeds_target = [0] * len(forecast_times)
        posterior_exceeds_target = [0] * len(forecast_times)
        has_target_data = False

    # Calculate true settlement for display
    from src.settlement.model import settlement_model, SoilParams
    params = SoilParams()
    true_settlement = settlement_model(times=np.array(all_times), params=params, cv=np.array([true_cv]))
    target_settlement = 0.067

    # Create subplot figure with 2 columns and secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Settlement Predictions", "Cv Distribution"),
        column_widths=[0.75, 0.25],
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )

    # LEFT PLOT - Settlement predictions
    # Observations
    if obs_times:
        fig.add_trace(
            go.Scatter(
                x=obs_times, y=settlement_obs,
                mode="markers", marker=dict(color="black", symbol="x", size=10),
                name="Observations", showlegend=True
            ), row=1, col=1, secondary_y=False
        )
        # Add vertical line at observation time
        fig.add_vline(x=max(obs_times), line=dict(color="black", dash="dash"), row=1, col=1)

    # Prior confidence interval (fill area)
    fig.add_trace(
        go.Scatter(
            x=forecast_times + forecast_times[::-1],
            y=prior_upper_quantile + prior_lower_quantile[::-1],
            fill="toself", fillcolor="rgba(0,0,255,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prior 90% CI", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Prior mean
    fig.add_trace(
        go.Scatter(
            x=forecast_times, y=prior_mean,
            mode="lines", line=dict(color="blue", width=2),
            name="Prior mean", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Posterior confidence interval (fill area)
    fig.add_trace(
        go.Scatter(
            x=forecast_times + forecast_times[::-1],
            y=posterior_upper_quantile + posterior_lower_quantile[::-1],
            fill="toself", fillcolor="rgba(255,0,0,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Posterior 90% CI", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Posterior mean
    fig.add_trace(
        go.Scatter(
            x=forecast_times, y=posterior_mean,
            mode="lines", line=dict(color="red", width=2),
            name="Posterior mean", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # True settlement
    fig.add_trace(
        go.Scatter(
            x=all_times, y=true_settlement.flatten(),
            mode="lines", line=dict(color="green", width=2),
            name="True settlement", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Target settlement line
    fig.add_hline(y=target_settlement, line=dict(color="black", width=2), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="black", width=2),
            name="Target settlement", showlegend=True
        ), row=1, col=1, secondary_y=False
    )

    # Target exceedance probabilities on secondary y-axis
    if has_target_data:
        fig.add_trace(
            go.Scatter(
                x=forecast_times, y=prior_exceeds_target,
                mode="lines", line=dict(color="blue", dash="dot", width=2),
                name="P(S>Target) Prior", yaxis="y2", showlegend=True
            ), row=1, col=1, secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_times, y=posterior_exceeds_target,
                mode="lines", line=dict(color="red", dash="dot", width=2),
                name="P(S>Target) Posterior", yaxis="y2", showlegend=True
            ), row=1, col=1, secondary_y=True
        )

    # RIGHT PLOT - Cv distributions (don"t show in main legend)
    fig.add_trace(
        go.Scatter(
            x=cv_grid, y=cv_prior_pdf,
            mode="lines", line=dict(color="blue", width=2),
            name="Prior PDF", showlegend=False
        ), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=cv_grid, y=cv_posterior_pdf,
            mode="lines", line=dict(color="red", width=2),
            name="Posterior PDF", showlegend=False
        ), row=1, col=2
    )

    # True Cv vertical line
    fig.add_vline(x=true_cv, line=dict(color="green", width=2), row=1, col=2)

    # Update axes
    fig.update_xaxes(title_text="Time [d]", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Settlement [m]", autorange="reversed", row=1, col=1, secondary_y=False, showgrid=True)
    if has_target_data:
        fig.update_yaxes(title_text="Target exceedance probability [%]", range=[0, 100], row=1, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Cv [m²/d]", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="Density [-]", row=1, col=2, showgrid=True)

    # Update layout with built-in legend positioned at top center, configured to wrap
    fig.update_layout(
        height=600,
        title=f"Settlement Analysis - Time: {time_key} days",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="center",
            y=-0.4,
            yanchor="bottom",
            itemwidth=140,
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10),
        )
    )

    fig.update_yaxes(range=[0, 0.1], row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 100], row=1, col=1, secondary_y=True)

    entries = [
        dict(color="blue", label="Prior PDF"),
        dict(color="red", label="Posterior PDF"),
        dict(color="green", label="True ${C}_{v}$"),
    ]

    y0 = 0.96
    dy = 0.08
    x_line0, x_line1 = 0.45, 0.60
    x_text = 0.63

    for i, e in enumerate(entries):
        y = y0 - i * dy
        # line sample
        fig.add_shape(
            type="line",
            x0=x_line0, y0=y, x1=x_line1, y1=y,
            xref="x2 domain", yref="y2 domain",
            line=dict(color=e["color"], width=3)
        )
        # label
        fig.add_annotation(
            x=x_text, y=y, xref="x2 domain", yref="y2 domain",
            text=e["label"], showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=3
        )

    return fig


def get_legend_parts(item):
    """Extract color, symbol, and text from legend item."""
    # Determine color based on item type
    if "Prior" in item and "P(S>" not in item:
        color = "blue"
    elif "Posterior" in item and "P(S>" not in item:
        color = "red"
    elif "True" in item:
        color = "green"
    elif "Target" in item:
        color = "black"
    elif "P(S>Target) Prior" in item:
        color = "blue"
    elif "P(S>Target) Posterior" in item:
        color = "red"
    elif "Observations" in item:
        color = "black"
    else:
        color = "black"

    # Extract symbol and text
    if "🞩" in item:
        symbol, text = item.split(" ", 1)
    elif item.startswith(("━", "█", "┅")):
        symbol = item[0]
        text = item[2:]
    else:
        symbol = "━"
        text = item

    return color, symbol, text


# Initialize the app
app = Dash(__name__)
app.title = "Settlement Analysis Visualization"

# Load predictions data
predictions_data = load_predictions_data()
available_times = sorted([int(k) for k in predictions_data.keys()])

# App layout
app.layout = html.Div([
    html.H1("Settlement Analysis Interactive Dashboard",
            style={"textAlign": "center", "marginBottom": 30}),

    html.Div([
        html.Label("Select Examination Time (days):",
                  style={"fontWeight": "bold", "marginBottom": 10}),
        dcc.Slider(
            id="time-slider",
            min=min(available_times),
            max=max(available_times),
            value=min(available_times),
            marks={time: {"label": str(time), "style": {"fontSize": "10px"}}
                   for i, time in enumerate(available_times) if i % 2 == 0},
            step=None,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={"margin": "20px", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),

    html.Div([
        dcc.Graph(id="settlement-plot", style={"height": "650px"})  # Increased height for legend
    ], style={"margin": "20px"}),

    html.Div([
        html.H4("About this Analysis:"),
        html.P([
            "This dashboard shows the evolution of settlement predictions over time. ",
            "The settlement analysis uses Bayesian updating to improve predictions as more observations become available."
        ]),
        html.Ul([
            html.Li("Blue lines show prior predictions (before incorporating observations)"),
            html.Li("Red lines show posterior predictions (after incorporating observations)"),
            html.Li("Green line shows the true settlement model"),
            html.Li("Black crosses show actual observations"),
            html.Li("Dotted lines show target exceedance probabilities (if available)"),
            html.Li("The right panel shows how the ${C}_{v}$ parameter distribution evolves")
        ])
    ], style={"margin": "20px", "padding": "20px", "backgroundColor": "#e9ecef", "borderRadius": "5px"})
])


@app.callback(
    Output("time-slider", "marks"),
    Input("time-slider", "value")
)
def update_slider_marks(selected_time):
    """Update slider marks to show all available times."""
    marks = {}
    for i, time in enumerate(available_times):
        if i % max(1, len(available_times) // 10) == 0 or time == selected_time:
            marks[time] = {"label": str(time), "style": {"fontSize": "10px"}}
    return marks


@app.callback(
    Output("settlement-plot", "figure"),
    Input("time-slider", "value")
)
def update_plot(selected_time):
    """Update the plot based on selected time."""
    # Find the closest available time
    closest_time = min(available_times, key=lambda x: abs(x - selected_time))
    
    # Create the figure
    fig = create_plotly_figure(predictions_data, closest_time)
    
    return fig


if __name__ == "__main__":

    app.run(debug=False)
