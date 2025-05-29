# tests/test_core_data_processing.py
import pytest
import pandas as pd
import geopandas as gpd
import os
from unittest.mock import patch, MagicMock
from config import app_config # For paths and settings

# Import functions to be tested
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    enrich_zone_geodata_with_health_aggregates,
    get_overall_kpis,
    get_chw_summary,
    get_patient_alerts_for_chw, # Assuming this exists or will be added
    get_clinic_summary,
    get_trend_data,
    get_supply_forecast_data,
    get_district_summary_kpis
)

# === Test load_health_records ===
@patch('utils.core_data_processing.os.path.exists')
@patch('utils.core_data_processing.pd.read_csv')
def test_load_health_records_success(mock_pd_read_csv, mock_os_exists, sample_health_records_df_main, mocker):
    mock_os_exists.return_value = True
    mock_pd_read_csv.return_value = sample_health_records_df_main.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')

    df = load_health_records()

    mock_pd_read_csv.assert_called_once_with(app_config.HEALTH_RECORDS_CSV, parse_dates=['date', 'referral_date', 'test_date'])
    assert not df.empty
    assert "ai_risk_score" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    mock_st_error.assert_not_called()

@patch('utils.core_data_processing.os.path.exists')
def test_load_health_records_file_not_found(mock_os_exists, mocker):
    mock_os_exists.return_value = False
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    df = load_health_records()
    assert df.empty
    mock_st_error.assert_called_once_with(f"Data file not found: {os.path.basename(app_config.HEALTH_RECORDS_CSV)}. Please check configuration.")

@patch('utils.core_data_processing.os.path.exists')
@patch('utils.core_data_processing.pd.read_csv')
def test_load_health_records_missing_required_cols(mock_pd_read_csv, mock_os_exists, mocker):
    mock_os_exists.return_value = True
    # Create a df missing a required column
    df_missing_cols = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'zone_id': ['Z001']}) # Missing patient_id, condition, ai_risk_score
    mock_pd_read_csv.return_value = df_missing_cols
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')

    df = load_health_records()
    assert df.empty
    # The exact message depends on which columns are checked first.
    # Example: "Health data is missing required columns: ['patient_id', 'condition', 'ai_risk_score']. Please check the CSV file."
    mock_st_error.assert_called_once()
    assert "missing required columns" in mock_st_error.call_args[0][0].lower()


# === Test load_zone_data ===
@patch('utils.core_data_processing.os.path.exists')
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.gpd.read_file')
def test_load_zone_data_success(mock_gpd_read_file, mock_pd_read_csv, mock_os_exists,
                                sample_zone_attributes_df_main, sample_zone_geometries_gdf_main, mocker):
    mock_os_exists.side_effect = [True, True] # First for attributes CSV, second for geometries GeoJSON
    mock_pd_read_csv.return_value = sample_zone_attributes_df_main.copy()
    mock_gpd_read_file.return_value = sample_zone_geometries_gdf_main.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')

    merged_gdf = load_zone_data()

    mock_pd_read_csv.assert_called_once_with(app_config.ZONE_ATTRIBUTES_CSV)
    mock_gpd_read_file.assert_called_once_with(app_config.ZONE_GEOMETRIES_GEOJSON)
    assert merged_gdf is not None
    assert not merged_gdf.empty
    assert 'name' in merged_gdf.columns and 'population' in merged_gdf.columns # From attributes
    assert 'geometry' in merged_gdf.columns # From geometries
    assert len(merged_gdf) == len(sample_zone_attributes_df_main) # Assuming all zone_ids match
    mock_st_error.assert_not_called()

@patch('utils.core_data_processing.os.path.exists')
def test_load_zone_data_attributes_file_not_found(mock_os_exists, mocker):
    mock_os_exists.side_effect = [False, True] # Attributes not found, Geometries found
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    gdf = load_zone_data()
    assert gdf is None
    mock_st_error.assert_called_once_with(f"Zone attributes file not found: {os.path.basename(app_config.ZONE_ATTRIBUTES_CSV)}.")

@patch('utils.core_data_processing.os.path.exists')
def test_load_zone_data_geometries_file_not_found(mock_os_exists, mocker):
    mock_os_exists.side_effect = [True, False] # Attributes found, Geometries not found
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    gdf = load_zone_data()
    assert gdf is None
    mock_st_error.assert_called_once_with(f"Zone geometries file not found: {os.path.basename(app_config.ZONE_GEOMETRIES_GEOJSON)}.")


# === Test enrich_zone_geodata_with_health_aggregates ===
def test_enrich_zone_geodata_with_health_aggregates_valid(sample_health_records_df_main, sample_zone_geometries_gdf_main, sample_zone_attributes_df_main):
    # First, create the base zone_gdf as load_zone_data would
    base_zone_gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    
    enriched_gdf = enrich_zone_geodata_with_health_aggregates(base_zone_gdf, sample_health_records_df_main)
    
    assert enriched_gdf is not None
    assert 'avg_risk_score' in enriched_gdf.columns
    assert 'active_cases' in enriched_gdf.columns
    assert 'prevalence_per_1000' in enriched_gdf.columns
    assert 'facility_coverage_score' in enriched_gdf.columns # Ensure this is calculated
    
    # Example check for a specific zone based on sample_health_records_df_main
    # ZN001: P001 (65), P003 (70), P001 (60), P007 (50) -> Avg Risk = (65+70+60+50)/4 = 61.25
    # ZN001: P001 (Hyp), P003 (Dia), P001 (Hyp), P007 (Dia) -> 0 active (TB/Malaria)
    # ZN001: Population 15000
    zn001_data = enriched_gdf[enriched_gdf['zone_id'] == 'ZN001'].iloc[0]
    assert zn001_data['avg_risk_score'] == pytest.approx(61.25)
    assert zn001_data['active_cases'] == 0
    assert zn001_data['prevalence_per_1000'] == pytest.approx(0)

    # ZN002: P002 (80, TB), P005 (75, TB), P002 (82, TB) -> Avg Risk = (80+75+82)/3 approx 79
    # ZN002: P002 (TB), P005 (TB), P002 (TB) -> 2 unique patients with TB
    # ZN002: Population 25000
    zn002_data = enriched_gdf[enriched_gdf['zone_id'] == 'ZN002'].iloc[0]
    assert zn002_data['avg_risk_score'] == pytest.approx((80+75+82)/3)
    assert zn002_data['active_cases'] == 2 # P002 and P005
    assert zn002_data['prevalence_per_1000'] == pytest.approx((2/25000)*1000)

def test_enrich_zone_geodata_empty_health_df(sample_zone_geometries_gdf_main, empty_health_df):
    # If health_df is empty, enrich should still return the zone_gdf, possibly with NaN/0 for new cols
    enriched = enrich_zone_geodata_with_health_aggregates(sample_zone_geometries_gdf_main, empty_health_df)
    assert enriched is not None
    assert 'avg_risk_score' in enriched.columns # Columns should be added
    assert enriched['avg_risk_score'].fillna(0).eq(0).all() # Values should be 0 or NaN then filled to 0


# === Test get_overall_kpis ===
def test_get_overall_kpis_valid(sample_health_records_df_main):
    kpis = get_overall_kpis(sample_health_records_df_main)
    assert kpis['total_patients'] == 8 # P001-P008
    # Active (TB/Malaria): P002 (TB), P004 (Malaria), P005 (TB), P006 (ARI-no)
    assert kpis['active_cases'] == 3 # P002, P004, P005
    assert kpis['pending_referrals'] == 3 # P001, P004, P008 (at latest date)
    assert kpis['avg_risk_score'] == pytest.approx(sample_health_records_df_main['ai_risk_score'].mean())
    # Positive tests in last 7 days (latest date is 2023-01-06)
    # P002 TB Positive on 2023-01-01
    assert kpis['positive_tests_past_week'] == 1

def test_get_overall_kpis_empty(empty_health_df):
    kpis = get_overall_kpis(empty_health_df)
    assert kpis['total_patients'] == 0
    assert kpis['avg_risk_score'] == 0


# === Test get_chw_summary ===
def test_get_chw_summary_valid(sample_health_records_df_main):
    # Assuming latest date is "today" for this summary
    latest_date = sample_health_records_df_main['date'].max()
    today_df = sample_health_records_df_main[sample_health_records_df_main['date'] == latest_date]
    summary = get_chw_summary(today_df) # Pass only today's data
    
    # On 2023-01-06: P008 (chw_visit=1)
    assert summary['visits_today'] == 1
    # Pending tasks: P008 (Pending referral) + P008 (TB traced = 0, but not TB) = 1
    # P001, P003, P004, P005, P006, P007 have different statuses or are not TB / not pending for referral
    # Need to filter sample_health_records_df_main for this carefully.
    # Total pending referrals: P001 (Hyp), P004 (Mal), P008 (Hyp) = 3
    # Total pending TB trace (TB condition & tb_contact_traced=0): P005 (but referral is Initiated, not pending for this test)
    # For this test, let's assume get_chw_summary considers all data for task counts:
    all_data_summary = get_chw_summary(sample_health_records_df_main) # Using all data for task counts logic
    assert all_data_summary['pending_tasks'] == 3 # P001, P004, P008 (Pending Referral) + 0 TB needing trace (P005 is initiated, P002 is completed/followed up)
    # High risk followups (risk >= 75 AND referral_status != Completed)
    # P002 (80, Completed), P005 (75, Initiated), P002 (82, Follow-up) -> P005, P002(Follow-up)
    assert all_data_summary['high_risk_followups'] == 2

# === Test get_patient_alerts_for_chw ===
# (This function was mentioned as new/specific; assuming its logic is similar to get_patient_alerts)
def test_get_patient_alerts_for_chw_valid(sample_health_records_df_main):
    # Assuming CHW alerts are for 'today' (latest date in data)
    latest_date_df = sample_health_records_df_main[sample_health_records_df_main['date'] == sample_health_records_df_main['date'].max()]
    alerts = get_patient_alerts_for_chw(latest_date_df, risk_threshold=app_config.RISK_THRESHOLDS['chw_alert_moderate'])
    assert isinstance(alerts, pd.DataFrame)
    if not alerts.empty:
        assert 'alert_reason' in alerts.columns
        # P008 on 2023-01-06 has risk 72 (>=65), referral Pending
        assert 'P008' in alerts['patient_id'].values

# === Test get_clinic_summary ===
def test_get_clinic_summary_valid(sample_health_records_df_main):
    summary = get_clinic_summary(sample_health_records_df_main)
    # Turnaround: (3+0+0)/3 = 1.0
    assert summary['avg_test_turnaround'] == pytest.approx(1.0)
    # Positive tests: P002 (Sputum Positive). Total tests with results: P002, P003, P007 (3 results)
    # Note: 'High' for Glucose is also treated as a "result" in some contexts.
    # If only 'Positive' for Sputum/RDT is counted: 1 positive / 3 actual tests with results = 33.3%
    # Let's refine get_clinic_summary logic if needed. Assuming current logic is:
    # Tested_df: P002 (Pos), P003(High), P004(Pend), P005(Pend), P007(Norm)
    # Dropna(subset=['test_result']): P002(Pos), P003(High), P004(Pend), P005(Pend), P007(Norm) (5) -> This is likely wrong in original.
    # Let's assume test_result != 'Pending' and != 'N/A' means a result.
    # P002(Pos), P003(High), P007(Norm) = 3 results. P002 is Positive. Rate = 1/3 = 33.3%
    assert summary['positive_test_rate'] == pytest.approx((1/3)*100)
    # Critical supply (<7 days): None in sample data
    assert summary['critical_supply_items'] == 0
    # Pending tests: P004 (RDT), P005 (Sputum) = 2
    assert summary['pending_tests_count'] == 2


# === Test get_trend_data ===
def test_get_trend_data_valid(sample_health_records_df_main):
    trend = get_trend_data(sample_health_records_df_main, 'ai_risk_score', period='D', agg_func='mean')
    assert isinstance(trend, pd.Series)
    assert not trend.empty
    assert pd.api.types.is_datetime64_any_dtype(trend.index)
    # 2023-01-01: (65+80)/2 = 72.5
    assert trend.loc[pd.to_datetime('2023-01-01')] == pytest.approx(72.5)

# === Test get_supply_forecast_data ===
def test_get_supply_forecast_data_valid(sample_health_records_df_main):
    forecast = get_supply_forecast_data(sample_health_records_df_main)
    assert isinstance(forecast, pd.DataFrame)
    if not forecast.empty: # If any supply items exist
        assert 'forecast_days' in forecast.columns
        assert 'lower_ci' in forecast.columns
        assert 'upper_ci' in forecast.columns
        assert not forecast.empty # sample_health_records_df_main has supply items

# === Test get_district_summary_kpis ===
def test_get_district_summary_kpis_valid(sample_health_records_df_main, sample_zone_geometries_gdf_main, sample_zone_attributes_df_main):
    base_zone_gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    enriched_gdf = enrich_zone_geodata_with_health_aggregates(base_zone_gdf, sample_health_records_df_main)
    
    if enriched_gdf is not None and not enriched_gdf.empty : # Check if enrichment was successful
        summary = get_district_summary_kpis(enriched_gdf)
        
        # Calculate expected values based on sample data and enrichment logic
        # This requires careful manual calculation based on how enrichment and get_district_summary_kpis work
        # Example: Avg Population Risk
        # ZN001: avg_risk 61.25, pop 15000 -> weighted_risk = 61.25 * 15000
        # ZN002: avg_risk (80+75+82)/3, pop 25000 -> weighted_risk = ((80+75+82)/3) * 25000
        # ZN003: avg_risk (55+40)/2, pop 18000 -> weighted_risk = 47.5 * 18000
        # ZN004: avg_risk 72, pop 22000 -> weighted_risk = 72 * 22000
        # Total pop = 15000+25000+18000+22000 = 80000
        # Expected_avg_pop_risk = ( (61.25*15000) + (((80+75+82)/3)*25000) + (47.5*18000) + (72*22000) ) / 80000
        # Manually calculate this value for assertion
        
        # For zones_high_risk (avg_risk_score >= 70 by default in app_config or current code)
        # ZN001 (61.25), ZN002 (79), ZN003 (47.5), ZN004 (72) -> ZN002 and ZN004 are high risk zones = 2
        assert summary['zones_high_risk'] == 2
        assert 'avg_population_risk' in summary # Check presence
        assert 'overall_facility_coverage' in summary # Check presence
    else:
        pytest.fail("Enrichment of GDF failed, cannot test get_district_summary_kpis properly.")
