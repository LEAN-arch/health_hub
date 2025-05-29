# tests/test_ui_visualization_helpers.py
import pytest
import plotly.graph_objects as go
import plotly.io as pio # For checking themes if needed
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Ensure correct import path and names from the overhauled utils
from utils.ui_visualization_helpers import (
    set_custom_plotly_theme, # Called on import, but test its effect
    render_kpi_card, render_traffic_light,
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_donut_chart,
    plot_heatmap,
    plot_layered_choropleth_map
)
from config import app_config # For default plot heights, map styles etc.

# Ensures the custom theme set in the module is active for tests that rely on theme defaults
# This should already be called when ui_visualization_helpers is imported.
# If not, uncomment: set_custom_plotly_theme()


# === Test Styled Components (render_kpi_card, render_traffic_light) ===
@patch('utils.ui_visualization_helpers.st.markdown') # Path to st.markdown within the tested module
def test_render_kpi_card_all_parameters(mock_st_markdown):
    render_kpi_card(
        title="Total Registered Patients", 
        value="1,234", 
        icon="🧑‍🤝‍🧑", 
        status="Moderate", # Test a standard status
        delta="+20 (Increase of 5%)", 
        delta_type="positive",
        help_text="Total unique patients registered in the system."
    )
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0] # Get the first arg of the first call
    assert '<div class="kpi-card moderate"' in html_output # Check for 'moderate' class
    assert 'title="Total unique patients registered in the system."' in html_output
    assert "Total Registered Patients" in html_output
    assert "1,234" in html_output
    assert "🧑‍🤝‍🧑" in html_output # Check for icon
    assert '<p class="kpi-delta positive">' in html_output # Check delta class and presence
    assert "+20 (Increase of 5%)" in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_kpi_card_semantic_status(mock_st_markdown):
    render_kpi_card(title="Coverage Score", value="95%", icon="✅", status="Good", delta_type="positive", delta="+2%")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    # The primary status class becomes 'good', semantic class is also 'status-good'
    # This tests if the CSS can handle combined classes like .kpi-card.good for specific styling
    assert '<div class="kpi-card good status-good"' in html_output
    assert "Coverage Score" in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_traffic_light_high_status_with_details(mock_st_markdown):
    render_traffic_light(message="Critical System Alert: High CPU Usage", status="High", details="CPU at 92%, Threshold 85%")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="traffic-light-indicator">' in html_output
    assert '<span class="traffic-light-dot status-high">' in html_output # Check for high status class
    assert "Critical System Alert: High CPU Usage" in html_output
    assert '<span class="traffic-light-details">CPU at 92%, Threshold 85%</span>' in html_output

# === Plotting Function Tests (using fixtures from the updated conftest.py) ===

# --- plot_annotated_line_chart Tests ---
def test_plot_annotated_line_chart_valid_data(sample_series_data):
    fig = plot_annotated_line_chart(sample_series_data, "Test Line Chart: Daily Count", y_axis_title="Daily Encounters")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1 # Base line trace + potential anomalies
    assert "Test Line Chart: Daily Count" in fig.layout.title.text
    assert fig.layout.yaxis.title.text == "Daily Encounters"
    assert fig.data[0].x[0] == sample_series_data.index[0] # Check data passthrough
    assert fig.data[0].y[0] == sample_series_data.values[0]
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT # Check default height from config

def test_plot_annotated_line_chart_empty_series():
    empty_series = pd.Series(dtype='float64', index=pd.to_datetime([]))
    fig = plot_annotated_line_chart(empty_series, "Empty Data Line Chart")
    assert isinstance(fig, go.Figure)
    assert "Empty Data Line Chart (No data available to display.)" in fig.layout.title.text
    assert len(fig.data) == 0 # No data traces should be added

def test_plot_annotated_line_chart_with_ci_and_target(sample_series_data):
    lower_ci = sample_series_data * 0.85
    upper_ci = sample_series_data * 1.15
    fig = plot_annotated_line_chart(sample_series_data, "Line Chart with CI & Target Line",
                                    target_line=14, target_label="Performance Target",
                                    show_ci=True, lower_bound_series=lower_ci, upper_bound_series=upper_ci)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2 # Main line + CI trace (and possibly anomalies)
    assert any(trace.name == "Confidence Interval" for trace in fig.data) # Check CI trace presence
    assert len(fig.layout.shapes) == 1 # For target line
    assert fig.layout.shapes[0].y0 == 14 # Target line Y value
    assert fig.layout.shapes[0].annotation.text == "Performance Target"


# --- plot_bar_chart Tests ---
def test_plot_bar_chart_valid_data(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Bar Chart: Category Values")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # One trace for non-grouped bar
    assert isinstance(fig.data[0], go.Bar)
    assert "Bar Chart: Category Values" in fig.layout.title.text
    assert fig.layout.xaxis.title.text == "Category"
    assert fig.layout.yaxis.title.text == "Value"
    # Note: px.bar might reorder categories, so direct list comparison of x-values can be fragile.
    # Instead, check if set of x-values matches.
    assert set(fig.data[0].x) == set(sample_bar_df['category'].unique())


def test_plot_bar_chart_grouped_data(sample_bar_df): # Uses sample_bar_df which has 'group' column
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Grouped Bar Chart by 'group'",
                         color_col='group', barmode='group') # Specify color_col and barmode
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == sample_bar_df['group'].nunique() # Number of traces = number of unique groups
    assert fig.layout.barmode == 'group'
    assert fig.layout.legend.title.text == 'Group' # Check legend title derived from color_col

# --- plot_donut_chart Tests ---
def test_plot_donut_chart_valid_data(sample_donut_df):
    fig = plot_donut_chart(sample_donut_df, labels_col='status', values_col='count', title="Donut Chart: Status Counts")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Pie)
    assert fig.data[0].hole is not None and 0.45 < fig.data[0].hole < 0.55 # Check typical donut hole size
    assert "Donut Chart: Status Counts" in fig.layout.title.text
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT + 40 # Check adjusted height

def test_plot_donut_chart_with_center_text(sample_donut_df):
    total_sum = sample_donut_df['count'].sum()
    fig = plot_donut_chart(sample_donut_df, labels_col='status', values_col='count', title="Donut with Center Text", center_text=f"Total: {total_sum}")
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == f"Total: {total_sum}"

# --- plot_heatmap Tests ---
def test_plot_heatmap_valid_matrix(sample_heatmap_df): # Fixture is a correlation-like matrix
    fig = plot_heatmap(sample_heatmap_df, "Heatmap: Correlation Matrix Style", colorscale="RdBu_r", zmid=0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert "Heatmap: Correlation Matrix Style" in fig.layout.title.text
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT + 100 # Adjusted height for heatmaps
    assert fig.data[0].zmin is None # zmin/zmax often auto-set by plotly unless explicitly for correlation scale (-1,1)
    assert fig.data[0].zmid == 0 # Check if zmid is passed for diverging colorscale


# --- plot_layered_choropleth_map Tests ---
def test_plot_layered_choropleth_map_valid_gdf(sample_choropleth_gdf): # sample_choropleth_gdf has 'zone_id', 'name', 'geometry', 'population', 'risk_score'
    fig = plot_layered_choropleth_map(
        gdf=sample_choropleth_gdf,
        value_col='risk_score',
        title="Choropleth Map: Zone Risk Scores",
        id_col='zone_id', # Matches 'zone_id' in sample_choropleth_gdf
        featureidkey_prefix='properties', # Default for GDFs where id_col is a property
        hover_cols=['name', 'population', 'risk_score'],
        mapbox_style="carto-positron" # Test a specific non-token style
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # Base choropleth trace
    assert isinstance(fig.data[0], go.Choroplethmapbox)
    assert "Choropleth Map: Zone Risk Scores" in fig.layout.title.text
    assert fig.layout.mapbox.style == "carto-positron" # Check if style is applied
    assert fig.layout.height == app_config.MAP_PLOT_HEIGHT


def test_plot_layered_choropleth_map_with_facilities(sample_choropleth_gdf):
    # Create a simple facility GDF for testing this feature
    facility_data = {
        'facility_id': ['Clinic A', 'Hospital B'],
        'capacity': [10, 50], # For sizing
        'type': ['Primary Care', 'Tertiary Hospital']
    }
    facility_geometry = [Point(0.5, 5.0), Point(15.0, 5.0)] # Points within typical extent of sample polygons
    facility_gdf_sample = gpd.GeoDataFrame(facility_data, geometry=facility_geometry, crs=app_config.DEFAULT_CRS)

    fig = plot_layered_choropleth_map(
        gdf=sample_choropleth_gdf, value_col='population', title="Choropleth with Health Facilities",
        id_col='zone_id', featureidkey_prefix='properties',
        facility_gdf=facility_gdf_sample,
        facility_size_col='capacity',
        facility_hover_name='facility_id',
        facility_color='#FF5733' # Custom facility color
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 # Choropleth trace + Scattermapbox for facilities
    assert isinstance(fig.data[1], go.Scattermapbox) # Check facility layer type
    assert fig.data[1].marker.color == '#FF5733' # Check custom facility color
    assert fig.data[1].name == 'Facilities Layer'

# General empty data / error case tests for plotting functions
@pytest.mark.parametrize("plot_function, req_cols_map", [
    (plot_annotated_line_chart, {'data_series': pd.Series(dtype=float)}), # Needs a Series
    (plot_bar_chart, {'df': pd.DataFrame(), 'x_col': 'X', 'y_col': 'Y'}),
    (plot_donut_chart, {'data_df': pd.DataFrame(), 'labels_col': 'L', 'values_col': 'V'}),
    (plot_heatmap, {'matrix_df': pd.DataFrame()}),
    (plot_layered_choropleth_map, {'gdf': gpd.GeoDataFrame(columns=['geometry'], crs=app_config.DEFAULT_CRS), 'value_col': 'Val', 'id_col':'id'})
])
def test_plotting_functions_empty_data(plot_function, req_cols_map):
    """Generic test for how plotting functions handle empty or minimal data."""
    title_test = "Empty Data Test Plot"
    fig = plot_function(title=title_test, **req_cols_map) # Pass necessary args
    assert isinstance(fig, go.Figure)
    # Expect a title indicating no data or an error message within the plot title
    assert f"{title_test} (No data" in fig.layout.title.text or f"{title_test} (Map Data Error" in fig.layout.title.text
    assert len(fig.data) == 0 # Usually no data traces are added for empty input
