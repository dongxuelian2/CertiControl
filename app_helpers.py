"""Presentation-only helpers for the Streamlit application."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go


def _complex_values(values: Iterable[complex]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.complex128).reshape(-1)


def eigenvalue_figure(eigenvalues: Iterable[complex], *, title: str = "Eigenvalues") -> go.Figure:
    values = _complex_values(eigenvalues)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.real(values),
            y=np.imag(values),
            mode="markers",
            name="poles",
            hovertemplate="Re=%{x:.6g}<br>Im=%{y:.6g}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash")
    fig.update_layout(
        title=title,
        xaxis_title="Real",
        yaxis_title="Imaginary",
        showlegend=True,
    )
    return fig


def open_closed_loop_figure(
    open_loop: Iterable[complex],
    closed_loop: Iterable[complex],
) -> go.Figure:
    open_values = _complex_values(open_loop)
    closed_values = _complex_values(closed_loop)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.real(open_values),
            y=np.imag(open_values),
            mode="markers",
            name="open-loop",
            hovertemplate="Open-loop<br>Re=%{x:.6g}<br>Im=%{y:.6g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.real(closed_values),
            y=np.imag(closed_values),
            mode="markers",
            name="closed-loop",
            hovertemplate="Closed-loop<br>Re=%{x:.6g}<br>Im=%{y:.6g}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash")
    fig.update_layout(
        title="Open-loop vs closed-loop poles",
        xaxis_title="Real",
        yaxis_title="Imaginary",
        showlegend=True,
    )
    return fig