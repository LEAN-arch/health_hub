import pytest
import pandas as pd
import geopandas as gpd
import os
from unittest.mock import patch, MagicMock # For mocking pd.read_csv, gpd.read_file

# Adjust import path if your tests directory is structured differently
# This assumes 'tests' is at the same level as 'utils'
from utils.data_processor import (
    load_health_data, load_geojson_data, calculate_incidence_rate, # Keep if you add it
    get_overall_kpis, merge_health_data_with_geojson
)


# --- Fixtures for Mock Data ---
@pytest.fixture
def sample_health_df_valid():
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'zone_id': ['ZN001', 'ZN002', 'ZN001'],
        'patient_id': ['P001', 'P002', 'P003'],
        'condition': ['Flu', 'TB', 'Flu'],
        'ai_risk_score': [60, 85, 70],
        'referral_status': ['Pending', 'Completed', 'Pending'],
        'test_result': ['N/A', 'Positive', 'Pending'],
        'test_date': pd.to_datetime([None, '2023-01-01', '2023-01-02']),
        'test_turnaround_days': [None, 3, None],
        'supply_level_days': [20, 10, 18],
        'tb_contact_traced': [0,1,0],
        'chw_visit': [1,1,0],
        'age': [45,30,62],
        'supply_item': ['MedA','MedB','MedA']
    })

@pytest.fixture
def sample_health_df_empty():
    return pd.DataFrame()

@pytest.fixture
def sample_health_df_missing_cols():
    return pd.DataFrame({ # Missing 'ai_risk_score'
        'date': pd.to_datetime(['2023-01-01']),
        'zone_id': ['ZN001'],
        'patient_id': ['P001'],
        'condition': ['Flu']
    })

@pytest.fixture
def sample_geojson_gdf_valid():
    # Create a simple GeoDataFrame
    data = {'zone_id': ['ZN001', 'ZN002'],
            'name': ['North', 'South'],
            'population': [1000, 1500]}
    # Create some dummy Point geometries for simplicity in test
    from shapely.geometry import Point
    geometry = [Point(0, 0), Point(1, 1)]
    return gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

# --- Mocking os.path.exists and file readers ---
@patch('utils.data_processor.os.path.exists')
@patch('utils.data_processor.pd.read_csv')
def test_load_health_data_valid(mock_read_csv, mock_exists, sample_health_df_valid, mocker):
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_health_df_valid.copy() # Return a copy
    mock_st_error = mocker.patch('utils.data_processor.st.error')
    
    df = load_health_data("dummy_path.csv")
    
    assert not df.empty
    assert "ai_risk_score" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    mock_read_csv.assert_called_once_with("data/dummy_path.csv", parse_dates=['date', 'referral_date', 'test_date'])
    mock_st_error.assert_not_called()

@patch('utils.data_processor.os.path.exists')
def test_load_health_data_file_not_found(mock_exists, mocker):
    mock_exists.return_value = False
    mock_st_error = mocker.patch('utils.data_processor.st.error')
    df = load_health_data("non_existent.csv")
    assert df.empty
    mock_st_error.assert_called_once_with("Data file not found: data/non_existent.csv. Please ensure it exists in the 'data' directory.")

@patch('utils.data_processor.os.path.exists')
@patch('utils.data_processor.pd.read_csv')
def test_load_health_data_missing_required_cols(mock_read_csv, mock_exists, sample_health_df_missing_cols, mocker):
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_health_df_missing_cols
    mock_st_error = mocker.patch('utils.data_processor.st.error')

    df = load_health_data("dummy_path.csv")
    assert df.empty
    mock_st_error.assert_called_once_with("Health data is missing required columns: ['ai_risk_score']. Please check the CSV file.")


@patch('utils.data_processor.os.path.exists')
@patch('utils.data_processor.gpd.read_file')
def test_load_geojson_data_valid(mock_read_file, mock_exists, sample_geojson_gdf_valid, mocker):
    mock_exists.return_value = True
    mock_read_file.return_value = sample_geojson_gdf_valid.copy()
    mock_st_error = mocker.patch('utils.data_processor.st.error')

    gdf = load_geojson_data("dummy_geo.json")
    assert gdf is not None
    assert not gdf.empty
    assert "population" in gdf.columns
    mock_read_file.assert_called_once_with("data/dummy_geo.json")
    mock_st_error.assert_not_called()

@patch('utils.data_processor.os.path.exists')
def test_load_geojson_data_file_not_found(mock_exists, mocker):
    mock_exists.return_value = False
    mock_st_error = mocker.patch('utils.data_processor.st.error')
    gdf = load_geojson_data("non_existent.geojson")
    assert gdf is None
    mock_st_error.assert_called_once_with("GeoJSON file not found: data/non_existent.geojson. Please ensure it exists in the 'data' directory.")


def test_get_overall_kpis_valid_data(sample_health_df_valid):
    kpis = get_overall_kpis(sample_health_df_valid)
    assert kpis["total_patients"] == 3
    assert kpis["active_cases"] == 1 # Only TB based on current logic
    assert kpis["pending_referrals"] == 2
    assert kpis["avg_risk_score"] == pytest.approx((60+85+70)/3)

def test_get_overall_kpis_empty_data(sample_health_df_empty):
    kpis = get_overall_kpis(sample_health_df_empty)
    assert kpis["total_patients"] == 0
    assert kpis["avg_risk_score"] == 0

def test_merge_health_data_with_geojson(sample_health_df_valid, sample_geojson_gdf_valid):
    merged_gdf = merge_health_data_with_geojson(sample_health_df_valid, sample_geojson_gdf_valid)
    assert merged_gdf is not None
    assert 'avg_risk_score' in merged_gdf.columns
    assert 'prevalence_per_1000' in merged_gdf.columns
    
    # Check values for a specific zone if data aligns
    zn001_data = merged_gdf[merged_gdf['zone_id'] == 'ZN001']
    if not zn001_data.empty:
        # Expected avg risk for ZN001 is (60+70)/2 = 65
        assert zn001_data.iloc[0]['avg_risk_score'] == pytest.approx(65)
    
    zn002_data = merged_gdf[merged_gdf['zone_id'] == 'ZN002']
    if not zn002_data.empty:
        assert zn002_data.iloc[0]['avg_risk_score'] == pytest.approx(85) # Only one patient
        # Prevalence: 1 active case (TB) in 1500 population = (1/1500)*1000
        # Our sample_health_df_valid has one TB case in ZN002
        # Our get_overall_kpis defines active cases as TB or Malaria
        # The merge_health_data counts active cases (TB or Malaria) for ZN002 as 1
        # sample_geojson_gdf_valid has ZN002 population 1500
        assert zn002_data.iloc[0]['prevalence_per_1000'] == pytest.approx((1/1500)*1000)


def test_merge_health_data_with_geojson_empty_health_data(sample_health_df_empty, sample_geojson_gdf_valid):
    merged_gdf = merge_health_data_with_geojson(sample_health_df_empty, sample_geojson_gdf_valid)
    # Should return the original geojson gdf as no health data to merge
    assert merged_gdf is not None
    assert merged_gdf.equals(sample_geojson_gdf_valid) # Or check specific properties

def test_merge_health_data_with_geojson_empty_geojson(sample_health_df_valid):
    empty_gdf = gpd.GeoDataFrame()
    merged_gdf = merge_health_data_with_geojson(sample_health_df_valid, empty_gdf)
    # Should return the empty geojson gdf
    assert merged_gdf is not None
    assert merged_gdf.empty

# Add more tests for other data_processor functions (get_chw_summary, get_clinic_summary, etc.)
# following similar patterns: provide valid and edge case inputs.
