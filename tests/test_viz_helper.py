import pytest
import plotly.graph_objects as go
from utils.viz_helper import plot_line_chart, plot_bar_chart, plot_donut_chart, plot_heatmap, plot_choropleth_map
import pandas as pd

def test_plot_line_chart():
    fig = plot_line_chart(["D-1", "D-2"], [1, 2], "Test Chart", "#3b82f6")
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure"

def test_plot_bar_chart():
    fig = plot_bar_chart(["A", "B"], [10, 20], "Test Bar", "#3b82f6")
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure"

def test_plot_donut_chart():
    fig = plot_donut_chart(["Positive", "Negative"], [10, 20], "Test Donut")
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure"

def test_plot_heatmap():
    matrix = pd.DataFrame([[1, 0.5], [0.5, 1]], index=["A", "B"], columns=["A", "B"])
    fig = plot_heatmap(matrix, "Test Heatmap")
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure"

def test_plot_choropleth_map():
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"zone": "A", "risk": 3}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}]}
    data = pd.DataFrame([{"zone": "A", "risk": 3}])
    fig = plot_choropleth_map(geojson, data, "zone", "risk", "Test Map")
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure"