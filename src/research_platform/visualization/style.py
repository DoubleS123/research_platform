"""Shared Plotly styling helpers."""

import plotly.graph_objects as go

BLUE = "#1f77b4"
LIGHT_BLUE = "#8ecae6"
ORANGE = "#ff7f0e"
DARK_BLUE = "#0b3d91"
GRAY = "#7f7f7f"
LIGHT_FILL = "rgba(31, 119, 180, 0.16)"


def apply_plot_style(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
) -> go.Figure:
    """Apply consistent clean styling for application plots."""
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=24, t=72, b=48),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 119, 180, 0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 119, 180, 0.08)")
    return fig
