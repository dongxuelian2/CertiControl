import importlib

import numpy as np

from app_helpers import eigenvalue_figure, open_closed_loop_figure


def test_app_module_imports_without_running_main():
    app = importlib.import_module("app")
    assert callable(app.main)


def test_open_loop_eigenvalue_figure_has_one_data_trace():
    fig = eigenvalue_figure([-1 + 1j, -1 - 1j])
    assert len(fig.data) == 1
    np.testing.assert_allclose(fig.data[0].x, [-1, -1])
    np.testing.assert_allclose(fig.data[0].y, [1, -1])


def test_open_closed_loop_figure_has_two_named_traces():
    fig = open_closed_loop_figure([1, -1], [-2, -3])
    assert len(fig.data) == 2
    assert fig.data[0].name == "open-loop"
    assert fig.data[1].name == "closed-loop"