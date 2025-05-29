import pytest
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from unittest.mock import patch

# Ensure correct import path and names
from utils.viz_helper import (
    set_custom_plotly_theme, # Call it to ensure theme is set for tests if not already by import
    render_kpi_card, render_traffic_light,
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_donut_chart,
    plot_heatmap,
    plot_layered_choropleth_map
)
import geopandas as gpd # For choropleth test data
from shapely.geometry import Point # For choropleth test data

# Apply the theme for testing context
set_custom_plotly_theme()

# --- Test Data Fixtures ---
@pytest.fixture
def sample_series_data():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    values = [10, 12, 11, 15]
    return pd.Series(values, index=dates, name="Metric Value")

@pytest.fixture
def sample_bar_df():
    return pd.DataFrame({
        'category': ['A', 'B', 'C'],
        'value': [100, 150, 80],
        'group': ['G1', 'G1', 'G2']
    })

@pytest.fixture
def sample_donut_df():
    return pd.DataFrame({
        'status': ['Positive', 'Negative', 'Pending'],
        'count': [20, 70, 10]
    })

@pytest.fixture
def sample_heatmap_df():
    return pd.DataFrame(
        np.random.rand(4, 4), # Random data for heatmap
        columns=[f'Var{i}' for i in range(1, 5)],
        index=[f'Feat{i}' for i in range(1, 5)]
    )
    
@pytest.fixture
def sample_choropleth_gdf():
    data = {'zone_id': ['Z1', 'Z2'],
            'name': ['Alpha', 'Beta'],
            'risk_score': [75.5, 60.2],
            'population': [1000, 1200]}
    geometry = [Point(0,0).buffer(0.5), Point(1,1).buffer(0.5)] # Simple polygon geometries
    return gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")


# --- Test render_kpi_card and render_traffic_light (Mock st.markdown) ---
@patch('utils.viz_helper.st.markdown')
def test_render_kpi_card(mock_st_markdown):
    render_kpi_card("Total Patients", "1,234", "🧑‍🤝‍🧑", status="Moderate", delta="+20", delta_type="positive")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="kpi-card status-moderate"' in html_output
    assert "Total Patients" in html_output
    assert "1,234" in html_output
    assert "🧑‍🤝‍🧑" in html_output
    assert '<p class="kpi-delta positive">+20</p>' in html_output

@patch('utils.viz_helper.st.markdown')
def test_render_traffic_light(mock_st_markdown):
    render_traffic_light("System Alert", "High", details="CPU > 90%")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="traffic-light-indicator">' in html_output
    assert '<span class="traffic-light-dot status-high">' in html_output
    assert "System Alert" in html_output
    assert '<span class="traffic-light-details">CPU > 90%</span>' in html_output


# --- Plotting Function Tests ---
def test_plot_annotated_line_chart_valid(sample_series_data):
    fig = plot_annotated_line_chart(sample_series_data, "Test Line Chart")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1 # Can be more if anomalies are detected
    assert fig.layout.title.text == "Test Line Chart"
    assert fig.data[0].x[0] == sample_series_data.index[0]
    assert fig.data[0].y[0] == sample_series_data.values[0]

def test_plot_annotated_line_chart_empty():
    empty_series = pd.Series(dtype='float64')
    fig = plot_annotated_line_chart(empty_series, "Empty Line Chart")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0 # Or checks specific to how empty charts are handled
    assert "Empty Line Chart (No data)" in fig.layout.title.text

def test_plot_annotated_line_chart_with_ci_and_target(sample_series_data):
    lower = sample_series_data * 0.9
    upper = sample_series_data * 1.1
    fig = plot_annotated_line_chart(sample_series_data, "Line with CI & Target", 
                                    target_line=12, target_label="Target",
                                    show_ci=True, lower_bound_series=lower, upper_bound_series=upper)
    assert len(fig.data) >= 2 # Main line + CI trace (plus anomalies if any)
    assert len(fig.layout.shapes) == 1 # For target line
    assert fig.layout.shapes[0].y0 == 12

def test_plot_bar_chart_valid(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, 'category', 'value', "Test Bar Chart")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Test Bar Chart"
    assert fig.data[0].x[0] == sample_bar_df['category'][0]

def test_plot_bar_chart_grouped(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, 'category', 'value', "Grouped Bar", color_col='group', barmode='group')
    assert isinstance(fig, go.Figure)
    # Number of traces depends on number of unique groups
    assert len(fig.data) == sample_bar_df['group'].nunique()

def test_plot_donut_chart_valid(sample_donut_df):
    fig = plot_donut_chart(sample_donut_df, 'status', 'count', "Test Donut Chart")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].hole > 0 # Check it's a donut
    assert fig.layout.title.text == "Test Donut Chart"

def test_plot_heatmap_valid(sample_heatmap_df):
    fig = plot_heatmap(sample_heatmap_df, "Test Heatmap")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert fig.layout.title.text == "Test Heatmap"

def test_plot_heatmap_non_numeric(mocker):
    # mock_st_error = mocker.patch("utils.viz_helper.st.error") # If you use st.error in helper
    df_non_numeric = pd.DataFrame({'A': ['x', 'y'], 'B': ['z', 'w']})
    fig = plot_heatmap(df_non_numeric, "Non-Numeric Heatmap")
    assert isinstance(fig, go.Figure)
    assert "Non-Numeric Heatmap (Contains non-numeric data)" in fig.layout.title.text
    # mock_st_error.assert_called_once() # if st.error is used

def test_plot_layered_choropleth_map_valid(sample_choropleth_gdf):
    fig = plot_layered_choropleth_map(sample_choropleth_gdf, 'risk_score', "Test Choropleth",
                                      id_col='zone_id', featureidkey_prop='zone_id',
                                      hover_cols=['name', 'population'])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # Base choropleth trace
    assert isinstance(fig.data[0], go.Choroplethmapbox) # Changed to mapbox
    assert fig.layout.title.text == "Test Choropleth"
    assert fig.layout.mapbox.style == "carto-positron"

def test_plot_layered_choropleth_map_with_facilities(sample_choropleth_gdf):
    # Create a simple facility GDF
    facility_data = {'facility_id': ['F1', 'F2'], 'capacity': [10,20]}
    facility_geometry = [Point(0.1, 0.1), Point(1.1, 1.1)]
    facility_gdf = gpd.GeoDataFrame(facility_data, geometry=facility_geometry, crs="EPSG:4326")

    fig = plot_layered_choropleth_map(sample_choropleth_gdf, 'risk_score', "Choropleth with Facilities",
                                      id_col='zone_id', featureidkey_prop='zone_id',
                                      facility_gdf=facility_gdf, facility_size_col='capacity', facility_hover_name='facility_id')
    assert len(fig.data) == 2 # Choropleth + Scattermapbox for facilities
    assert isinstance(fig.data[1], go.Scattermapbox)

# Add more tests for empty data, invalid columns for each plot type.
