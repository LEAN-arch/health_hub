# utils/core_data_processing.py
import pandas as pd
import geopandas as gpd
import os
import logging
import streamlit as st
from config import app_config # Import the new config

# Configure logging using settings from app_config
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600) # Cache for 1 hour
def load_health_records():
    """
    Load and validate main health CSV data from the path specified in app_config.
    """
    file_path = app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Health records file not found: {file_path}")
            st.error(f"Data file not found: {os.path.basename(file_path)}. Please check configuration and ensure the file exists in 'data_sources/'.")
            return pd.DataFrame()

        df = pd.read_csv(file_path, parse_dates=['date', 'referral_date', 'test_date'])

        required_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score"] # Core minimum
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns in health records: {missing}")
            st.error(f"Health records data is missing required columns: {missing}. Please check the CSV file content and headers.")
            return pd.DataFrame()

        # Basic type conversions and cleaning for important columns
        numeric_cols = ['age', 'chw_visit', 'test_turnaround_days', 'supply_level_days', 'ai_risk_score', 'tb_contact_traced']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure 'referral_status' and 'test_result' are strings to handle 'N/A' consistently
        for col in ['referral_status', 'test_result', 'test_type', 'supply_item', 'gender', 'condition', 'zone_id', 'patient_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown') # Fill NaNs as 'Unknown' string

        logger.info(f"Successfully loaded and validated health records from {file_path} with {len(df)} rows.")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Health records file is empty: {file_path}")
        st.error(f"The health records data file '{os.path.basename(file_path)}' is empty.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading health records from {file_path}: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred while loading health records: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_zone_data():
    """
    Loads zone attribute data from CSV and zone geometries from GeoJSON, then merges them.
    Uses paths from app_config. Returns a GeoDataFrame.
    """
    attributes_path = app_config.ZONE_ATTRIBUTES_CSV
    geometries_path = app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"Attempting to load zone attributes from: {attributes_path}")
    logger.info(f"Attempting to load zone geometries from: {geometries_path}")

    try:
        # Load attributes
        if not os.path.exists(attributes_path):
            logger.error(f"Zone attributes file not found: {attributes_path}")
            st.error(f"Zone attributes file '{os.path.basename(attributes_path)}' not found. Please check configuration.")
            return None
        zone_attributes_df = pd.read_csv(attributes_path)
        required_attr_cols = ["zone_id", "name", "population", "socio_economic_index", "num_clinics", "avg_travel_time_clinic_min"]
        if not all(col in zone_attributes_df.columns for col in required_attr_cols):
            missing_attr = [col for col in required_attr_cols if col not in zone_attributes_df.columns]
            logger.error(f"Zone attributes CSV missing required columns: {missing_attr}")
            st.error(f"Zone attributes CSV missing required columns: {missing_attr}. Please check file content.")
            return None

        # Load geometries
        if not os.path.exists(geometries_path):
            logger.error(f"Zone geometries file not found: {geometries_path}")
            st.error(f"Zone geometries file '{os.path.basename(geometries_path)}' not found. Please check configuration.")
            return None
        zone_geometries_gdf = gpd.read_file(geometries_path)
        if "zone_id" not in zone_geometries_gdf.columns:
            logger.error("Zone geometries GeoJSON missing 'zone_id' property in features.")
            st.error("Zone geometries GeoJSON missing 'zone_id' property. This is required for merging.")
            return None

        # Standardize zone_id type for robust merging
        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str)
        zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str)

        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left")

        if merged_gdf['name'].isnull().any():
            unmatched_geometries = merged_gdf[merged_gdf['name'].isnull()]['zone_id'].tolist()
            logger.warning(f"Some zone geometries could not be matched with attributes (zone_ids: {unmatched_geometries}). These zones will have missing attribute data.")

        logger.info(f"Successfully loaded and merged zone data. Resulting GeoDataFrame has {len(merged_gdf)} features.")
        return merged_gdf

    except pd.errors.EmptyDataError as ede:
        logger.error(f"A zone data file is empty: {ede}")
        st.error(f"A zone data file is empty. Please check '{os.path.basename(attributes_path)}' and '{os.path.basename(geometries_path)}'.")
        return None
    except Exception as e:
        logger.error(f"Error loading zone data: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred while loading zone data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def enrich_zone_geodata_with_health_aggregates(zone_gdf, health_df):
    """
    Enriches the zone GeoDataFrame with aggregated health metrics from health_df.
    """
    if health_df.empty:
        logger.warning("Health records DataFrame is empty. Cannot enrich zone geodata significantly.")
        default_cols = ['avg_risk_score', 'active_cases', 'prevalence_per_1000', 'facility_coverage_score', 'total_tests_conducted', 'chw_visits_in_zone']
        for col in default_cols:
            if col not in zone_gdf.columns: zone_gdf[col] = 0
        return zone_gdf
    if zone_gdf is None or zone_gdf.empty:
        logger.warning("Zone GeoDataFrame is empty or None. Cannot enrich.")
        return zone_gdf

    logger.info(f"Enriching zone geodata ( {len(zone_gdf)} zones) with health records ({len(health_df)} records).")

    health_df_copy = health_df.copy()
    zone_gdf_copy = zone_gdf.copy()

    health_df_copy['zone_id'] = health_df_copy['zone_id'].astype(str)
    zone_gdf_copy['zone_id'] = zone_gdf_copy['zone_id'].astype(str)

    zone_summary = health_df_copy.groupby('zone_id').agg(
        avg_risk_score=('ai_risk_score', 'mean'),
        active_cases=('patient_id', lambda x: health_df_copy.loc[x.index]['condition'].astype(str).isin(['TB', 'Malaria', 'ARI']).nunique()),
        total_tests_conducted=('test_type', lambda x: health_df_copy.loc[x.index]['test_type'].astype(str).nunique(dropna=False)),
        chw_visits_in_zone=('chw_visit', 'sum')
    ).reset_index()

    enriched_gdf = zone_gdf_copy.merge(zone_summary, on='zone_id', how='left')

    for col in ['avg_risk_score', 'active_cases', 'total_tests_conducted', 'chw_visits_in_zone']:
        if col in enriched_gdf.columns:
            enriched_gdf[col] = enriched_gdf[col].fillna(0)

    if 'population' in enriched_gdf.columns and 'active_cases' in enriched_gdf.columns:
        enriched_gdf['population'] = pd.to_numeric(enriched_gdf['population'], errors='coerce').fillna(0)
        enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(
            lambda row: (row['active_cases'] / row['population']) * 1000 if row['population'] > 0 else 0, axis=1
        )
    else:
        enriched_gdf['prevalence_per_1000'] = 0.0

    if 'avg_travel_time_clinic_min' in enriched_gdf.columns and 'num_clinics' in enriched_gdf.columns:
        enriched_gdf['avg_travel_time_clinic_min'] = pd.to_numeric(enriched_gdf['avg_travel_time_clinic_min'], errors='coerce')
        enriched_gdf['num_clinics'] = pd.to_numeric(enriched_gdf['num_clinics'], errors='coerce')

        min_travel = enriched_gdf['avg_travel_time_clinic_min'].min()
        max_travel = enriched_gdf['avg_travel_time_clinic_min'].max()
        if pd.isna(min_travel) or pd.isna(max_travel) or max_travel == min_travel:
            enriched_gdf['travel_score'] = 50.0
        else:
            enriched_gdf['travel_score'] = 100 * (1 - (enriched_gdf['avg_travel_time_clinic_min'] - min_travel) / (max_travel - min_travel))

        min_clinics = enriched_gdf['num_clinics'].min()
        max_clinics = enriched_gdf['num_clinics'].max()
        if pd.isna(min_clinics) or pd.isna(max_clinics) or max_clinics == min_clinics:
            enriched_gdf['clinic_count_score'] = 50.0
        else:
            enriched_gdf['clinic_count_score'] = 100 * (enriched_gdf['num_clinics'] - min_clinics) / (max_clinics - min_clinics)

        enriched_gdf['facility_coverage_score'] = (enriched_gdf['travel_score'].fillna(50) * 0.6 + enriched_gdf['clinic_count_score'].fillna(50) * 0.4)
    else:
        logger.warning("Missing 'avg_travel_time_clinic_min' or 'num_clinics' for facility coverage calculation.")
        enriched_gdf['facility_coverage_score'] = 50.0

    for score_col in ['travel_score', 'clinic_count_score', 'facility_coverage_score', 'prevalence_per_1000']:
        if score_col in enriched_gdf.columns:
             enriched_gdf[score_col] = enriched_gdf[score_col].fillna(enriched_gdf[score_col].median() if pd.api.types.is_numeric_dtype(enriched_gdf[score_col]) and not enriched_gdf[score_col].empty and enriched_gdf[score_col].notna().any() else 0)


    logger.info("Zone geodata successfully enriched with health aggregates.")
    return enriched_gdf

# --- KPI Calculation Functions ---
@st.cache_data
def get_overall_kpis(df_health_records):
    if df_health_records.empty:
        return {"total_patients": 0, "active_cases": 0, "avg_risk_score": 0, "pending_referrals": 0, "positive_tests_past_week": 0}

    df_copy = df_health_records.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy.dropna(subset=['date'], inplace=True)
    if df_copy.empty: return {"total_patients": 0, "active_cases": 0, "avg_risk_score": 0, "pending_referrals": 0, "positive_tests_past_week": 0}

    latest_date = df_copy['date'].max()
    one_week_ago = latest_date - pd.Timedelta(days=7)

    total_patients = df_copy['patient_id'].astype(str).nunique()
    active_conditions = ['TB', 'Malaria', 'ARI']
    active_cases = df_copy[df_copy['condition'].astype(str).isin(active_conditions)]['patient_id'].astype(str).nunique()
    
    avg_risk_score_series = pd.to_numeric(df_copy['ai_risk_score'], errors='coerce')
    avg_risk_score = avg_risk_score_series.mean() if not avg_risk_score_series.empty and avg_risk_score_series.notna().any() else 0
    
    pending_referrals = df_copy[df_copy['referral_status'].astype(str) == 'Pending']['patient_id'].astype(str).nunique()

    df_copy['test_date'] = pd.to_datetime(df_copy['test_date'], errors='coerce')
    positive_tests = df_copy[
        (df_copy['test_result'].astype(str) == 'Positive') & (df_copy['test_date'] >= one_week_ago)
    ]['patient_id'].astype(str).nunique()

    return {
        "total_patients": total_patients, "active_cases": active_cases,
        "avg_risk_score": avg_risk_score, "pending_referrals": pending_referrals,
        "positive_tests_past_week": positive_tests
    }

@st.cache_data
def get_chw_summary(df_chw_view):
    if df_chw_view.empty:
        return {"visits_today": 0, "pending_tasks": 0, "high_risk_followups": 0}

    df_copy = df_chw_view.copy()
    # Ensure relevant columns are of expected types
    df_copy['chw_visit'] = pd.to_numeric(df_copy['chw_visit'], errors='coerce').fillna(0)
    df_copy['referral_status'] = df_copy['referral_status'].astype(str)
    df_copy['condition'] = df_copy['condition'].astype(str)
    df_copy['tb_contact_traced'] = pd.to_numeric(df_copy['tb_contact_traced'], errors='coerce').fillna(0)
    df_copy['ai_risk_score'] = pd.to_numeric(df_copy['ai_risk_score'], errors='coerce')


    visits_today = df_copy[df_copy['chw_visit'] == 1].shape[0]

    pending_referrals = df_copy[df_copy['referral_status'] == 'Pending'].shape[0]
    pending_tb_trace = df_copy[(df_copy['condition'] == 'TB') & (df_copy['tb_contact_traced'] == 0)].shape[0]
    pending_tasks = pending_referrals + pending_tb_trace

    high_risk_followups = df_copy[
        (df_copy['ai_risk_score'].notna()) &
        (df_copy['ai_risk_score'] >= app_config.RISK_THRESHOLDS['high']) &
        (~df_copy['referral_status'].isin(['Completed', 'Closed', 'Unknown'])) # Not completed/closed/unknown
    ].shape[0]

    return {"visits_today": visits_today, "pending_tasks": pending_tasks, "high_risk_followups": high_risk_followups}

@st.cache_data
def get_patient_alerts_for_chw(df_chw_view, risk_threshold=None):
    if df_chw_view.empty: return pd.DataFrame()
    
    df_copy = df_chw_view.copy()
    # Ensure types and handle NaNs
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy['test_date'] = pd.to_datetime(df_copy['test_date'], errors='coerce')
    df_copy.dropna(subset=['date'], inplace=True)
    if df_copy.empty: return pd.DataFrame()

    df_copy['ai_risk_score'] = pd.to_numeric(df_copy['ai_risk_score'], errors='coerce') # Keep NaNs for now
    for col in ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id']:
        if col in df_copy.columns:
             df_copy[col] = df_copy[col].astype(str).fillna('Unknown')
    df_copy['tb_contact_traced'] = pd.to_numeric(df_copy['tb_contact_traced'], errors='coerce').fillna(0)


    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['chw_alert_moderate']
    latest_view_date = df_copy['date'].max()
    recent_test_cutoff = latest_view_date - pd.Timedelta(days=3)

    alert_conditions = (
        (df_copy['ai_risk_score'].notna() & (df_copy['ai_risk_score'] >= risk_thresh)) |
        ((df_copy['test_result'] == 'Positive') & (df_copy['test_date'].notna()) & (df_copy['test_date'] >= recent_test_cutoff)) |
        (df_copy['referral_status'] == 'Pending') |
        ((df_copy['condition'] == 'TB') & (df_copy['tb_contact_traced'] == 0))
    )
    alerts_df = df_copy[alert_conditions].copy()

    if alerts_df.empty: return pd.DataFrame()

    def determine_reason_chw(row):
        if pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= risk_thresh: return f"High Risk ({row['ai_risk_score']:.0f})"
        if row['test_result'] == 'Positive' and pd.notna(row['test_date']) and row['test_date'] >= recent_test_cutoff: return f"Positive Test ({row['condition']})"
        if row['referral_status'] == 'Pending': return f"Pending Referral ({row['condition']})"
        if row['condition'] == 'TB' and row['tb_contact_traced'] == 0: return "TB Contact Tracing"
        return "Other"

    alerts_df['alert_reason'] = alerts_df.apply(determine_reason_chw, axis=1)
    alerts_df = alerts_df.sort_values(by=['ai_risk_score', 'date'], ascending=[False, False]) # Higher risk, more recent
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)

    output_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'date']
    existing_output_cols = [col for col in output_cols if col in alerts_df.columns]
    return alerts_df[existing_output_cols]


@st.cache_data
def get_clinic_summary(df_clinic_view):
    if df_clinic_view.empty:
        return {"avg_test_turnaround": 0.0, "positive_test_rate": 0.0, "critical_supply_items": 0, "pending_tests_count": 0}

    df_copy = df_clinic_view.copy()
    df_copy['test_turnaround_days'] = pd.to_numeric(df_copy['test_turnaround_days'], errors='coerce')
    df_copy['test_result'] = df_copy['test_result'].astype(str)
    df_copy['supply_level_days'] = pd.to_numeric(df_copy['supply_level_days'], errors='coerce')
    df_copy['supply_item'] = df_copy['supply_item'].astype(str)

    valid_turnaround_df = df_copy.dropna(subset=['test_turnaround_days'])
    avg_turnaround = valid_turnaround_df['test_turnaround_days'].mean() if not valid_turnaround_df.empty else 0.0

    conclusive_tests_df = df_copy[~df_copy['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])]
    positive_tests = conclusive_tests_df[conclusive_tests_df['test_result'] == 'Positive'].shape[0]
    total_conclusive_tests = conclusive_tests_df.shape[0]
    positive_rate = (positive_tests / total_conclusive_tests) * 100 if total_conclusive_tests > 0 else 0.0

    critical_supply = df_copy[
        (df_copy['supply_level_days'].notna()) &
        (df_copy['supply_level_days'] <= app_config.CRITICAL_SUPPLY_DAYS) &
        (~df_copy['supply_item'].isin(['Unknown', 'N/A', 'nan']))
    ]['supply_item'].nunique()
    
    pending_tests_count = df_copy[df_copy['test_result'] == 'Pending'].shape[0]

    return {
        "avg_test_turnaround": avg_turnaround, "positive_test_rate": positive_rate,
        "critical_supply_items": critical_supply, "pending_tests_count": pending_tests_count
    }

@st.cache_data
def get_patient_alerts_for_clinic(df_clinic_view, risk_threshold=None):
    if df_clinic_view.empty: return pd.DataFrame()

    df_copy = df_clinic_view.copy()
    # Ensure types and handle NaNs
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy['test_date'] = pd.to_datetime(df_copy['test_date'], errors='coerce')
    df_copy.dropna(subset=['date'], inplace=True)
    if df_copy.empty: return pd.DataFrame()

    df_copy['ai_risk_score'] = pd.to_numeric(df_copy['ai_risk_score'], errors='coerce')
    for col in ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id']:
         if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).fillna('Unknown')

    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['moderate']
    latest_view_date = df_copy['date'].max()
    recent_test_cutoff = latest_view_date - pd.Timedelta(days=7)

    alert_conditions = (
        (df_copy['ai_risk_score'].notna() & (df_copy['ai_risk_score'] >= risk_thresh)) |
        ((df_copy['test_result'] == 'Positive') & (df_copy['test_date'].notna()) & (df_copy['test_date'] >= recent_test_cutoff)) |
        ((df_copy['test_result'] == 'Pending') & (df_copy['condition'].isin(['TB','Malaria']))) # Use simple 'in' for string
    )
    alerts_df = df_copy[alert_conditions].copy()

    if alerts_df.empty: return pd.DataFrame()

    def determine_reason_clinic(row):
        # Order of checks matters for precedence
        if pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= risk_thresh:
            return f"High Risk ({row['ai_risk_score']:.0f})"
        if row['test_result'] == 'Positive' and pd.notna(row['test_date']) and row['test_date'] >= recent_test_cutoff:
            return f"Recent Positive Test ({row['condition']})"
        if row['test_result'] == 'Pending' and row['condition'] in ['TB','Malaria']: # Corrected here
            return f"Pending Critical Test ({row['condition']})"
        return "Review Case"

    alerts_df['alert_reason'] = alerts_df.apply(determine_reason_clinic, axis=1)
    alerts_df = alerts_df.sort_values(by=['ai_risk_score', 'date'], ascending=[False, False])
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)

    output_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result', 'referral_status', 'alert_reason', 'date']
    existing_output_cols = [col for col in output_cols if col in alerts_df.columns]
    return alerts_df[existing_output_cols]


@st.cache_data
def get_trend_data(df, value_col, date_col='date', period='D', agg_func='mean'):
    if df.empty or value_col not in df.columns or date_col not in df.columns:
        logger.warning(f"Cannot generate trend for '{value_col}': DataFrame empty or columns missing.")
        return pd.Series(dtype='float64').rename_axis(date_col if date_col in df.columns else 'date')

    df_copy = df.copy()
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        # For value_col, ensure it's numeric if we're doing mean/sum
        if agg_func in ['mean', 'sum']:
            df_copy[value_col] = pd.to_numeric(df_copy[value_col], errors='coerce')
        
        df_copy.dropna(subset=[date_col, value_col], inplace=True)

        if df_copy.empty:
             logger.warning(f"No valid data points left for trend '{value_col}' after NaT/NaN drop.")
             return pd.Series(dtype='float64').rename_axis(date_col)

        if agg_func == 'count':
            trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].count()
        elif agg_func == 'sum':
            trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].sum()
        elif agg_func == 'nunique':
             trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].nunique()
        else: # Default to mean
            trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].mean()

        return trend.fillna(0)
    except Exception as e:
        logger.error(f"Error generating trend data for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype='float64').rename_axis(date_col if date_col in df.columns else 'date')

@st.cache_data
def get_supply_forecast_data(df_health_records, forecast_days_out=7):
    if df_health_records.empty: return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])

    df_copy = df_health_records.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy['supply_item'] = df_copy['supply_item'].astype(str).fillna('Unknown')
    df_copy['supply_level_days'] = pd.to_numeric(df_copy['supply_level_days'], errors='coerce')

    # Get the latest supply level for each item, excluding 'Unknown' items
    latest_supplies = df_copy[~df_copy['supply_item'].isin(['Unknown', 'N/A', 'nan'])].sort_values('date').drop_duplicates(subset=['supply_item'], keep='last')
    latest_supplies = latest_supplies[latest_supplies['supply_level_days'].notna() & (latest_supplies['supply_level_days'] > 0)]

    if latest_supplies.empty: return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])

    forecast_list = []
    for _, row in latest_supplies.iterrows():
        item = row['supply_item']
        current_level_days = row['supply_level_days']
        last_recorded_date = row['date']

        if pd.isna(last_recorded_date): continue # Skip if date is invalid

        daily_consumption = 1.0 # Simplistic: assumes level_days means it depletes in that many days

        for i in range(1, forecast_days_out + 1):
            forecast_date = last_recorded_date + pd.Timedelta(days=i)
            forecasted_level = max(0, current_level_days - (daily_consumption * i))

            ci_factor = 0.15
            lower_ci = max(0, forecasted_level * (1 - ci_factor))
            upper_ci = forecasted_level * (1 + ci_factor)

            forecast_list.append({
                'date': forecast_date, 'item': item, 'forecast_days': forecasted_level,
                'lower_ci': lower_ci, 'upper_ci': upper_ci
            })

    return pd.DataFrame(forecast_list) if forecast_list else pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])

@st.cache_data
def get_district_summary_kpis(enriched_zone_gdf):
    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
        return {"avg_population_risk": 0.0, "overall_facility_coverage": 0.0, "zones_high_risk": 0}

    gdf_copy = enriched_zone_gdf.copy()
    # Ensure necessary columns are numeric and handle NaNs before calculations
    for col in ['population', 'avg_risk_score', 'facility_coverage_score']:
        if col in gdf_copy.columns:
            gdf_copy[col] = pd.to_numeric(gdf_copy[col], errors='coerce').fillna(0)
        else: # If column doesn't exist, add it with zeros to prevent KeyErrors
            gdf_copy[col] = 0.0


    total_population = gdf_copy['population'].sum()
    avg_pop_risk = 0.0
    if 'avg_risk_score' in gdf_copy.columns and total_population > 0:
        avg_pop_risk = (gdf_copy['avg_risk_score'] * gdf_copy['population']).sum() / total_population

    overall_facility_coverage = 0.0
    if 'facility_coverage_score' in gdf_copy.columns and total_population > 0:
        overall_facility_coverage = (gdf_copy['facility_coverage_score'] * gdf_copy['population']).sum() / total_population

    zones_high_risk = 0
    if 'avg_risk_score' in gdf_copy.columns:
        zones_high_risk = gdf_copy[gdf_copy['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]

    return {
        "avg_population_risk": avg_pop_risk,
        "overall_facility_coverage": overall_facility_coverage,
        "zones_high_risk": zones_high_risk
    }
