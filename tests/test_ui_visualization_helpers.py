# tests/test_ui_visualization_helpers.py
import pytest
import plotly.graph_objects as go
import plotly.io as pio # For checking themes if needed
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Ensure correct import path and names
from utils.ui_visualization_helpers import (
    set_custom_plotly_theme,
    render_kpi_card, render_traffic_light,
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_donut_chart,
    plot_heatmap,
    plot_layered_choropleth_map
)
from config import app_config # For default plot heights etc.

# Apply the theme for testing context if not already done by module import
# This ensures that any theme defaults used in the functions are active
set_custom_plotly_theme()

# === Test render_kpi_card and render_traffic_light (Mock st.markdown) ===
@patch('utils.ui_visualization_helpers.st.markdown') # Path to st within the tested module
def test_render_kpi_card_all_params(mock_st_markdown):
    render_kpi_card(
        title="Total Patients", 
        value="1,234", 
        icon="🧑‍🤝‍🧑", 
        status="Moderate", 
        delta="+20 (5%)", 
        delta_type="positive",
        help_text="Total unique patients registered."
    )
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="kpi-card status-moderate"' in html_output
    assert 'title="Total unique patients registered."' in html_output
    assert "Total Patients" in html_output
    assert "1,234" in html_output
    assert "🧑‍🤝‍🧑" in html_output
    assert '<p class="kpi-delta positive">+20 (5%)</p>' in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_kpi_card_minimal_params(mock_st_markdown):
    render_kpi_card(title="Active Alerts", value="5", icon="🚨")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="kpi-card "' in html_output # No status class
    assert 'title=""' not in html_output # No help text
    assert "Active Alerts" in html_output
    assert '<p class="kpi-delta' not in html_output # No delta

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_traffic_light_all_params(mock_st_markdown):
    render_traffic_light(message="System Alert: High CPU", status="High", details="CPU usage at 92%")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="traffic-light-indicator">' in html_output
    assert '<span class="traffic-light-dot status-high">' in html_output
    assert "System Alert: High CPU" in html_output
    assert '<span class="traffic-light-details">CPU usage at 92%</span>' in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_traffic_light_no_details(mock_st_markdown):
    render_traffic_light(message="Zone A: Moderate Risk", status="Moderate")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<span class="traffic-light-dot status-moderate">' in html_output
    assert "Zone A: Moderate Risk" in html_output
    assert '<span class="traffic-light-details">' not in html_output


# === Plotting Function Tests (using fixtures from conftest.py) ===

def test_plot_annotated_line_chart_valid(sample_series_data): # Fixture from conftest.py
    fig = plot_annotated_line_chart(sample_series_data, "Test Line Chart", y_axis_title="Count")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1 # Base line trace + potential anomalies
    assert fig.layout.title.text == "Test Line Chart"
    assert fig.layout.yaxis.title.text == "Count"
    assert fig.data[0].x[0] == sample_series_data.index[0] # Check data passthrough
    assert fig.data[0].y[0] == sample_series_data.values[0]
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT # Check default height

def test_plot_annotated_line_chart_empty():
    empty_series = pd.Series(dtype='float64', index=pd.to_datetime([]))
    fig = plot_annotated_line_chart(empty_series, "Empty Line Chart")
    assert isinstance(fig, go.Figure)
    assert "Empty Line Chart (No data)" in fig.layout.title.text
    assert len(fig.data) == 0

def test_plot_annotated_line_chart_with_ci_and_target(sample_series_data):
    lower = sample_series_data * 0.9
    upper = sample_series_data * 1.1
    fig = plot_annotated_line_chart(sample_series_data, "Line with CI & Target", 
                                    target_line=12, target_label="Target Value",
                                    show_ci=True, lower_bound_series=lower, upper_bound_series=upper)
    assert len(fig.data) >= 2 # Main line + CI trace (plus anomalies if any)
    assert fig.data[1].fill == "toself" # Check CI trace properties
    assert len(fig.layout.shapes) == 1 # For target line
    assert fig.layout.shapes[0].y0 == 12
    assert fig.layout.shapes[0].annotation.text == "Target Value"


def test_plot_bar_chart_valid(sample_bar_df): # Fixture from conftest.py
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Test Bar Chart")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)
    assert fig.layout.title.text == "Test Bar Chart"
    assert fig.layout.xaxis.title.text == "Category"
    assert fig.layout.yaxis.title.text == "Value"
    assert list(fig.data[0].x) == sample_bar_df['category'].tolist()

def test_plot_bar_chart_empty(empty_health_df): # Use an empty df that might have expected cols
    fig = plot_bar_chart(empty_health_df, x_col='zone_id', y_col='ai_risk_score', title="Empty Bar Chart")
    assert isinstance(fig, go.Figure)
    assert "Empty Bar Chart (No data)" in fig.layout.title.text
    assert len(fig.data) == 0

def test_plot_bar_chart_grouped(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Grouped Bar", color_col='group', barmode='group')
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == sample_bar_df['group'].nunique() # Number of traces should match unique groups
    assert fig.layout.barmode == 'group'


def test_plot_donut_chart_valid(sample_donut_df): # Fixture from conftest.py
    fig = plot_donut_chart(sample_donut_df, labels_col='status', values_col='count', title="Test Donut Chart")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Pie)
    assert fig.data[0].hole > 0.4 and fig.data[0].hole < 0.5 # Check donut hole size
    assert fig.layout.title.text == "Test Donut Chart"
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT + 30 # Adjusted height for donut

def test_plot_donut_chart_empty(empty_health_df):
    fig = plot_donut_chart(empty_health_df, labels_col='condition', values_col='patient_id', title="Empty Donut")
    assert isinstance(fig, go.Figure)
    assert "Empty Donut (No data)" in fig.layout.title.text
    assert len(fig.data) == 0


def test_plot_heatmap_valid(sample_heatmap_df): # Fixture from conftest.py
    fig = plot_heatmap(sample_heatmap_df, "Test Heatmap", height=450)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert fig.layout.title.text == "Test Heatmap"
    assert fig.layout.height == 450
    assert list(fig.data[0].x) == sample_heatmap_df.columns.tolist()

def test_plot_heatmap_empty(empty_health_df): # An empty df, not necessarily a matrix
    fig = plot_heatmap(empty_health_df, "Empty Heatmap")
    assert isinstance(fig, go.Figure)
    assert "Empty Heatmap (No data or invalid data)" in fig.layout.title.text
    assert len(fig.data) == 0

def test_plot_heatmap_non_numeric():
    df_non_numeric = pd.DataFrame({'A': ['x', 'y', 'z'], 'B': ['p', 'q', 'r'], 'C': [1, 's', 3]})
    fig = plot_heatmap(df_non_numeric, "Non-Numeric Heatmap")
    assert isinstance(fig, go.Figure)
    assert "Non-Numeric Heatmap (Contains non-numeric data)" in fig.layout.title.text
    assert len(fig.data) == 0

def test_plot_heatmap_correlation_style(sample_heatmap_df): # Assuming it's a correlation matrix
    # Make it look like a correlation matrix (values between -1 and 1, diagonal is 1)
    corr_df = sample_heatmap_df.corr() # Make it a correlation matrix
    fig = plot_heatmap(corr_df, "Correlation Heatmap Test")
    assert fig.data[0].zmin == -1
    assert fig.data[0].zmax == 1
    assert fig.data[0].colorscale == "RdYlGn_r"


def test_plot_layered_choropleth_map_valid(sample_choropleth_gdf): # Fixture from conftest.py
    fig = plot_layered_choropleth_map(
        sample_choropleth_gdf, 
        value_col='risk_score', 
        title="Test Choropleth Map",
        id_col='zone_id', # Matches sample_choropleth_gdf
        featureidkey_prop='zone_id', # Assumes GeoJSON features have "properties":{"zone_id": ...}
        hover_cols=['name', 'population'],
        height=app_config.MAP_PLOT_HEIGHT
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # Base choropleth trace
    assert isinstance(fig.data[0], go.Choroplethmapbox)
    assert fig.layout.title.text == "Test Choropleth Map"
    assert fig.layout.mapbox.style == "carto-positron"
    assert fig.layout.height == app_config.MAP_PLOT_HEIGHT

def test_plot_layered_choropleth_map_empty(empty_gdf):
    fig = plot_layered_choropleth_map(empty_gdf, 'risk_score', "Empty Choropleth")
    assert isinstance(fig, go.Figure)
    assert "Empty Choropleth (No data or invalid GeoDataFrame)" in fig.layout.title.text
    assert len(fig.data) == 0

def test_plot_layered_choropleth_map_with_facilities(sample_choropleth_gdf):
    # Create a simple facility GDF for testing
    facility_data = {'facility_id': ['F1', 'F2'], 'capacity': [10, 20], 'type': ['Clinic', 'Hospital']}
    facility_geometry = [Point(0.05, 0.05), Point(1.05, 1.05)] # Points within/near the sample polygons
    facility_gdf = gpd.GeoDataFrame(facility_data, geometry=facility_geometry, crs="EPSG:4326")

    fig = plot_layered_choropleth_map(
        sample_choropleth_gdf, 
        value_col='risk_score', 
        title="Choropleth with Facilities",
        id_col='zone_id', featureidkey_prop='zone_id',
        facility_gdf=facility_gdf, 
        facility_size_col='capacity', 
        facility_hover_name='facility_id'
    )
    assert len(fig.data) == 2 # Choropleth + Scattermapbox for facilities
    assert isinstance(fig.data[1], go.Scattermapbox)
    assert fig.data[1].marker.size is not None # Check if size is applied (even if it's default)


# Add tests for invalid columns, mismatched data, etc. for each plot type
# For example:
def test_plot_bar_chart_invalid_column(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, x_col='non_existent_col', y_col='value', title="Invalid Column Bar")
    assert isinstance(fig, go.Figure)
    assert "Invalid Column Bar (No data)" in fig.layout.title.text
    assert len(fig.data) == 0
