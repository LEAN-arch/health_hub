# tests/test_core_data_processing.py
import pytest
import pandas as pd
import geopandas as gpd
import os
import numpy as np
from unittest.mock import patch, MagicMock
from config import app_config # For paths and settings

# Import functions to be tested
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    load_iot_clinic_environment_data, # Added test for this
    enrich_zone_geodata_with_health_aggregates,
    get_overall_kpis,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_clinic_summary,
    get_clinic_environmental_summary, # Added test
    get_patient_alerts_for_clinic, # Added test
    get_trend_data,
    get_supply_forecast_data,
    get_district_summary_kpis
)

# === Test load_health_records ===
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_health_records_success(mock_os_exists, mock_pd_read_csv, sample_health_records_df_main, mocker):
    mock_os_exists.return_value = True
    mock_pd_read_csv.return_value = sample_health_records_df_main.copy() # Use the rich fixture
    mock_st_error = mocker.patch('utils.core_data_processing.st.error') # Mock streamlit error display

    df = load_health_records()

    mock_pd_read_csv.assert_called_once_with(app_config.HEALTH_RECORDS_CSV, low_memory=False)
    assert not df.empty
    assert "ai_risk_score" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert pd.api.types.is_numeric_dtype(df['ai_risk_score']) # Check numeric conversion
    assert df['condition'].dtype == 'object' or pd.api.types.is_string_dtype(df['condition']) # Check string conversion
    mock_st_error.assert_not_called()

@patch('utils.core_data_processing.os.path.exists')
def test_load_health_records_file_not_found(mock_os_exists, mocker):
    mock_os_exists.return_value = False
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    df = load_health_records() # Should return an empty DataFrame
    assert df.empty
    mock_st_error.assert_called_once()
    assert f"Data file '{os.path.basename(app_config.HEALTH_RECORDS_CSV)}' not found" in mock_st_error.call_args[0][0]

@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_health_records_missing_critical_cols(mock_os_exists, mock_pd_read_csv, mocker):
    mock_os_exists.return_value = True
    # Create a df missing a critical column like 'patient_id'
    df_missing_critical = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'zone_id': ['Z001'], 'condition':['TB'], 'ai_risk_score':[70]})
    mock_pd_read_csv.return_value = df_missing_critical
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')

    df = load_health_records()
    assert df.empty # Should return empty if critical cols are missing
    mock_st_error.assert_called_once()
    assert "missing critical data in columns: ['patient_id']" in mock_st_error.call_args[0][0]


# === Test load_zone_data ===
@patch('utils.core_data_processing.gpd.read_file')
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_zone_data_success(mock_os_exists, mock_pd_read_csv, mock_gpd_read_file,
                                sample_zone_attributes_df_main, sample_zone_geometries_gdf_main, mocker):
    mock_os_exists.return_value = True # Both files exist
    mock_pd_read_csv.return_value = sample_zone_attributes_df_main.copy()
    mock_gpd_read_file.return_value = sample_zone_geometries_gdf_main.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')

    merged_gdf = load_zone_data()

    mock_pd_read_csv.assert_called_once_with(app_config.ZONE_ATTRIBUTES_CSV)
    mock_gpd_read_file.assert_called_once_with(app_config.ZONE_GEOMETRIES_GEOJSON)
    assert isinstance(merged_gdf, gpd.GeoDataFrame)
    assert not merged_gdf.empty
    assert 'name' in merged_gdf.columns and 'population' in merged_gdf.columns # From attributes
    assert 'geometry' in merged_gdf.columns # From geometries
    assert merged_gdf.crs.to_string().upper() == app_config.DEFAULT_CRS.upper() # Check CRS standardization
    mock_st_error.assert_not_called()

# Add tests for file not found for zone_attributes and zone_geometries similar to load_health_records

# === Test load_iot_clinic_environment_data ===
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_iot_data_success(mock_os_exists, mock_pd_read_csv, sample_iot_clinic_df_main, mocker):
    mock_os_exists.return_value = True
    mock_pd_read_csv.return_value = sample_iot_clinic_df_main.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    df = load_iot_clinic_environment_data()
    assert not df.empty
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    assert 'zone_id' in df.columns # Check derived/mapped zone_id
    mock_st_error.assert_not_called()

@patch('utils.core_data_processing.os.path.exists')
def test_load_iot_data_file_not_found(mock_os_exists, mocker): # Added this test
    mock_os_exists.return_value = False
    # For IoT, missing file returns an empty DataFrame, not None and doesn't call st.error directly
    # It logs a warning. We can't easily check st.info/st.warning here without more complex mocking.
    df = load_iot_clinic_environment_data()
    assert df.empty # Should return the default_empty_iot_df


# === Test enrich_zone_geodata_with_health_aggregates ===
def test_enrich_zone_geodata_with_health_aggregates_valid_data(
    sample_zone_geometries_gdf_main, sample_zone_attributes_df_main,
    sample_health_records_df_main, sample_iot_clinic_df_main
):
    # Create base_gdf as it would be loaded by load_zone_data
    base_gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    
    enriched_gdf = enrich_zone_geodata_with_health_aggregates(
        base_gdf, sample_health_records_df_main, sample_iot_clinic_df_main
    )
    
    assert isinstance(enriched_gdf, gpd.GeoDataFrame)
    assert not enriched_gdf.empty
    # Check for key aggregated columns
    assert 'avg_risk_score' in enriched_gdf.columns
    assert 'active_tb_cases' in enriched_gdf.columns
    assert 'prevalence_per_1000' in enriched_gdf.columns
    assert 'facility_coverage_score' in enriched_gdf.columns
    assert 'zone_avg_co2' in enriched_gdf.columns # From IoT data
    
    # Example check for ZoneA based on sample data
    zone_a_data = enriched_gdf[enriched_gdf['zone_id'] == 'ZoneA'].iloc[0]
    # Expected avg_risk for ZoneA P001(85),P003(30),P005(88),P008(92),P001(80 -> uses this as latest for P001), P011(15 from conftest was for PapSmear not full record)
    # P001: 80 (latest) | P003: 30 | P005: 88 | P008: 92
    # From fixture: P001 (85, 80), P003 (30), P005 (88), P008 (92) -> In ZoneA.
    # In health_records.csv data: P001(85),P003(30),P005(88),P008(92),P001(80) from records on/before 10-07 for P001, ...
    # P001 latest in sample is 80. P003 is 30. P005 is 88. P008 is 92.
    # Patients in ZoneA: P001, P003, P005, P008.
    # Avg risk for these specific latest records = (80+30+88+92)/4 = 72.5
    # The logic groups by zone_id and takes mean of ai_risk_score for all records in that zone.
    # Zone A records AI scores: 85, 30, 88, 92, 80.  Mean = (85+30+88+92+80)/5 = 75.0
    assert zone_a_data['avg_risk_score'] == pytest.approx(75.0) 
    assert zone_a_data['active_tb_cases'] >= 1 # P001, P005 are TB in ZoneA

    # Check IoT aggregation
    assert pd.notna(zone_a_data['zone_avg_co2']) # ZoneA has IoT data in sample_iot_clinic_df_main
    # ZoneA in iot_df: C01 records: 650, 700, 680, 660. Latest room: WA (700), C1 (680)
    # The enrichment takes latest per room THEN averages these for the zone. (700+680)/2 = 690
    assert zone_a_data['zone_avg_co2'] == pytest.approx((700+680)/2) # Avg of 700 (WA), 680 (C1) for ZoneA latest


# === Test get_overall_kpis ===
def test_get_overall_kpis_with_valid_data(sample_health_records_df_main):
    # Assume date_filter_end is latest date in data
    latest_date = sample_health_records_df_main['date'].max()
    kpis = get_overall_kpis(sample_health_records_df_main, date_filter_end=latest_date)
    assert kpis['total_patients'] > 0
    assert isinstance(kpis['avg_patient_risk'], float)
    assert kpis['active_tb_cases_current'] >= 0
    assert kpis['malaria_rdt_positive_rate_period'] >= 0.0


# === Test get_chw_summary ===
def test_get_chw_summary_with_valid_day_data(sample_health_records_df_main):
    # Use data for a single day from the sample
    single_day_df = sample_health_records_df_main[sample_health_records_df_main['date'] == pd.to_datetime('2023-10-01')].copy()
    summary = get_chw_summary(single_day_df)
    assert summary['visits_today'] >= 0
    assert summary['tb_contacts_to_trace_today'] >= 0
    assert isinstance(summary['avg_patient_risk_visited_today'], float)


# === Test get_patient_alerts_for_chw ===
def test_get_patient_alerts_for_chw_valid_day_data(sample_health_records_df_main):
    single_day_df = sample_health_records_df_main[sample_health_records_df_main['date'] == pd.to_datetime('2023-10-03')].copy()
    alerts = get_patient_alerts_for_chw(single_day_df, risk_threshold_moderate=app_config.RISK_THRESHOLDS['chw_alert_moderate'])
    assert isinstance(alerts, pd.DataFrame)
    if not alerts.empty:
        assert 'alert_reason' in alerts.columns
        assert 'priority_score' in alerts.columns


# === Test get_clinic_summary ===
def test_get_clinic_summary_with_period_data(sample_health_records_df_main):
    # Use a period of data
    start_date = pd.to_datetime('2023-10-01'); end_date = pd.to_datetime('2023-10-05')
    period_df = sample_health_records_df_main[(sample_health_records_df_main['date'] >= start_date) & (sample_health_records_df_main['date'] <= end_date)].copy()
    summary = get_clinic_summary(period_df)
    assert summary['tb_sputum_positivity'] >= 0.0
    assert summary['malaria_positivity'] >= 0.0
    assert summary['avg_test_turnaround_all_tests'] >= 0.0
    assert summary['key_drug_stockouts'] >= 0


# === Test get_clinic_environmental_summary ===
def test_get_clinic_environmental_summary_valid_period(sample_iot_clinic_df_main):
    start_date = pd.to_datetime('2023-10-01'); end_date = pd.to_datetime('2023-10-01') # Single day iot
    period_iot_df = sample_iot_clinic_df_main[(sample_iot_clinic_df_main['timestamp'] >= start_date) & (sample_iot_clinic_df_main['timestamp'] <= end_date)].copy()
    summary = get_clinic_environmental_summary(period_iot_df)
    assert isinstance(summary['avg_co2_overall'], float)
    assert summary['rooms_co2_alert_latest'] >= 0


# === Test get_patient_alerts_for_clinic ===
def test_get_patient_alerts_for_clinic_valid_period(sample_health_records_df_main):
    start_date = pd.to_datetime('2023-10-01'); end_date = pd.to_datetime('2023-10-05')
    period_df = sample_health_records_df_main[(sample_health_records_df_main['date'] >= start_date) & (sample_health_records_df_main['date'] <= end_date)].copy()
    alerts = get_patient_alerts_for_clinic(period_df)
    assert isinstance(alerts, pd.DataFrame)
    if not alerts.empty:
        assert 'alert_reason' in alerts.columns
        assert 'priority_score' in alerts.columns


# === Test get_trend_data ===
def test_get_trend_data_with_valid_data(sample_health_records_df_main):
    trend = get_trend_data(sample_health_records_df_main, 'ai_risk_score', period='D', agg_func='mean')
    assert isinstance(trend, pd.Series)
    assert not trend.empty
    assert pd.api.types.is_datetime64_any_dtype(trend.index)
    # Example check for a specific date based on sample_health_records_df_main
    # On 2023-10-01: P001 (85), P002 (70) -> Mean = (85+70)/2 = 77.5
    assert trend.loc[pd.to_datetime('2023-10-01')] == pytest.approx(77.5)

# === Test get_supply_forecast_data ===
def test_get_supply_forecast_data_with_valid_data(sample_health_records_df_main):
    forecast = get_supply_forecast_data(sample_health_records_df_main, forecast_days_out=7)
    assert isinstance(forecast, pd.DataFrame)
    if not forecast.empty: # If any supply items to forecast
        assert 'forecast_days' in forecast.columns
        assert 'lower_ci' in forecast.columns # Check for CI columns
        assert 'upper_ci' in forecast.columns
        assert 'estimated_stockout_date' in forecast.columns


# === Test get_district_summary_kpis ===
def test_get_district_summary_kpis_with_enriched_gdf(sample_enriched_gdf_main): # Use the new enriched fixture
    if sample_enriched_gdf_main is None or sample_enriched_gdf_main.empty:
        pytest.skip("Skipping district KPI test as enriched GDF fixture is empty or None.")

    summary = get_district_summary_kpis(sample_enriched_gdf_main)
    assert isinstance(summary['avg_population_risk'], float)
    assert summary['zones_high_risk_count'] >= 0
    assert summary['district_tb_burden_total'] >= 0
    assert 'population_weighted_avg_steps' in summary # New KPI


# === Tests for Empty or Invalid Inputs for Robustness ===
def test_get_overall_kpis_empty_df(empty_health_df_with_schema):
    kpis = get_overall_kpis(empty_health_df_with_schema)
    assert kpis['total_patients'] == 0
    assert kpis['avg_patient_risk'] == 0.0

def test_get_trend_data_empty_df(empty_health_df_with_schema):
    trend = get_trend_data(empty_health_df_with_schema, 'ai_risk_score')
    assert trend.empty

def test_enrich_zone_geodata_empty_health_df(sample_zone_geometries_gdf_main, sample_zone_attributes_df_main, empty_health_df_with_schema):
    base_gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    enriched = enrich_zone_geodata_with_health_aggregates(base_gdf, empty_health_df_with_schema, None)
    assert isinstance(enriched, gpd.GeoDataFrame)
    assert 'avg_risk_score' in enriched.columns
    assert enriched['avg_risk_score'].fillna(0).eq(0).all() # Aggregated health values should be 0 or NaN, then filled to 0

def test_enrich_zone_geodata_empty_base_gdf(empty_gdf_with_schema, sample_health_records_df_main):
    enriched = enrich_zone_geodata_with_health_aggregates(empty_gdf_with_schema, sample_health_records_df_main, None)
    assert isinstance(enriched, gpd.GeoDataFrame)
    assert enriched.empty # Enriching an empty GDF should result in an empty (or schema-only) GDF

# Add more tests for other functions with empty/invalid inputs if time permits...
