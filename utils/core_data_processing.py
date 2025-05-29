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
        for col in ['referral_status', 'test_result', 'test_type', 'supply_item', 'gender', 'condition']:
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
        
        if merged_gdf['name'].isnull().any(): # Check if merge failed for some zones
            unmatched_geometries = merged_gdf[merged_gdf['name'].isnull()]['zone_id'].tolist()
            logger.warning(f"Some zone geometries could not be matched with attributes (zone_ids: {unmatched_geometries}). These zones will have missing attribute data.")
            # Consider how to handle this - for now, they will have NaNs for attribute columns.

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
        # Add default aggregate columns to zone_gdf to prevent downstream errors
        for col in ['avg_risk_score', 'active_cases', 'prevalence_per_1000', 'facility_coverage_score']:
            if col not in zone_gdf.columns: zone_gdf[col] = 0
        return zone_gdf
    if zone_gdf is None or zone_gdf.empty:
        logger.warning("Zone GeoDataFrame is empty or None. Cannot enrich.")
        return zone_gdf

    logger.info(f"Enriching zone geodata ( {len(zone_gdf)} zones) with health records ({len(health_df)} records).")

    # Standardize zone_id type for robust merging
    health_df['zone_id'] = health_df['zone_id'].astype(str)
    zone_gdf['zone_id'] = zone_gdf['zone_id'].astype(str)

    # Aggregate health data by zone_id
    zone_summary = health_df.groupby('zone_id').agg(
        avg_risk_score=('ai_risk_score', 'mean'),
        # Count unique patients with specified active conditions
        active_cases=('patient_id', lambda x: health_df.loc[x.index]['condition'].isin(['TB', 'Malaria', 'ARI']).nunique()),
        total_tests_conducted=('test_type', lambda x: health_df.loc[x.index]['test_type'].nunique(dropna=False)), # Example new metric
        chw_visits_in_zone=('chw_visit', 'sum') # Example new metric
    ).reset_index()
    
    enriched_gdf = zone_gdf.merge(zone_summary, on='zone_id', how='left')
    
    # Fill NaNs that result from zones with no matching health records
    for col in ['avg_risk_score', 'active_cases', 'total_tests_conducted', 'chw_visits_in_zone']:
        if col in enriched_gdf.columns:
            enriched_gdf[col] = enriched_gdf[col].fillna(0)
    
    # Calculate prevalence per 1000 using GeoJSON population
    if 'population' in enriched_gdf.columns and 'active_cases' in enriched_gdf.columns:
        # Ensure population is not zero to avoid division by zero
        enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(
            lambda row: (row['active_cases'] / row['population']) * 1000 if row['population'] > 0 else 0, axis=1
        )
    else:
        enriched_gdf['prevalence_per_1000'] = 0
    
    # Calculate facility coverage score
    if 'avg_travel_time_clinic_min' in enriched_gdf.columns and 'num_clinics' in enriched_gdf.columns:
        # Min-Max normalization for scores (0-100)
        # Travel time: lower is better (invert by 1 - normalized_value)
        min_travel = enriched_gdf['avg_travel_time_clinic_min'].min()
        max_travel = enriched_gdf['avg_travel_time_clinic_min'].max()
        if max_travel == min_travel: # Avoid division by zero
            enriched_gdf['travel_score'] = 50.0 # Neutral score
        else:
            enriched_gdf['travel_score'] = 100 * (1 - (enriched_gdf['avg_travel_time_clinic_min'] - min_travel) / (max_travel - min_travel))

        # Num clinics: higher is better
        min_clinics = enriched_gdf['num_clinics'].min()
        max_clinics = enriched_gdf['num_clinics'].max()
        if max_clinics == min_clinics:
            enriched_gdf['clinic_count_score'] = 50.0
        else:
            enriched_gdf['clinic_count_score'] = 100 * (enriched_gdf['num_clinics'] - min_clinics) / (max_clinics - min_clinics)
        
        # Weighted average for facility coverage score
        enriched_gdf['facility_coverage_score'] = (enriched_gdf['travel_score'] * 0.6 + enriched_gdf['clinic_count_score'] * 0.4)
    else:
        logger.warning("Missing 'avg_travel_time_clinic_min' or 'num_clinics' for facility coverage calculation.")
        enriched_gdf['facility_coverage_score'] = 50.0 # Default neutral score

    # Fill any remaining NaNs in score columns with a neutral value or 0
    for score_col in ['travel_score', 'clinic_count_score', 'facility_coverage_score', 'prevalence_per_1000']:
        if score_col in enriched_gdf.columns:
            enriched_gdf[score_col] = enriched_gdf[score_col].fillna(enriched_gdf[score_col].median() if not enriched_gdf[score_col].empty else 0)


    logger.info("Zone geodata successfully enriched with health aggregates.")
    return enriched_gdf

# --- KPI Calculation Functions ---
@st.cache_data
def get_overall_kpis(df_health_records):
    if df_health_records.empty:
        return {"total_patients": 0, "active_cases": 0, "avg_risk_score": 0, "pending_referrals": 0, "positive_tests_past_week": 0}
    
    latest_date = df_health_records['date'].max()
    one_week_ago = latest_date - pd.Timedelta(days=7)

    total_patients = df_health_records['patient_id'].nunique()
    active_conditions = ['TB', 'Malaria', 'ARI'] # Example definition of conditions that count as "active"
    active_cases = df_health_records[df_health_records['condition'].isin(active_conditions)]['patient_id'].nunique()
    avg_risk_score = df_health_records['ai_risk_score'].mean() if not df_health_records['ai_risk_score'].empty else 0
    pending_referrals = df_health_records[df_health_records['referral_status'] == 'Pending']['patient_id'].nunique()
    
    positive_tests = df_health_records[
        (df_health_records['test_result'] == 'Positive') & (df_health_records['test_date'] >= one_week_ago)
    ]['patient_id'].nunique()

    return {
        "total_patients": total_patients, "active_cases": active_cases, 
        "avg_risk_score": avg_risk_score, "pending_referrals": pending_referrals,
        "positive_tests_past_week": positive_tests
    }

@st.cache_data
def get_chw_summary(df_chw_view): # df_chw_view is typically data for a single day or CHW's scope
    if df_chw_view.empty:
        return {"visits_today": 0, "pending_tasks": 0, "high_risk_followups": 0}

    visits_today = df_chw_view[df_chw_view['chw_visit'] == 1].shape[0]
    
    pending_referrals = df_chw_view[df_chw_view['referral_status'] == 'Pending'].shape[0]
    # Example: TB contacts needing tracing (specific to TB cases where contact tracing isn't marked complete)
    pending_tb_trace = df_chw_view[(df_chw_view['condition'] == 'TB') & (df_chw_view['tb_contact_traced'] == 0)].shape[0]
    pending_tasks = pending_referrals + pending_tb_trace # Simplified sum of distinct task types
    
    high_risk_followups = df_chw_view[
        (df_chw_view['ai_risk_score'] >= app_config.RISK_THRESHOLDS['high']) &
        (~df_chw_view['referral_status'].isin(['Completed', 'Closed'])) # Not completed or closed
    ].shape[0]
    
    return {"visits_today": visits_today, "pending_tasks": pending_tasks, "high_risk_followups": high_risk_followups}

@st.cache_data
def get_patient_alerts_for_chw(df_chw_view, risk_threshold=None):
    if df_chw_view.empty: return pd.DataFrame()
    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['chw_alert_moderate']
    
    # Criteria for CHW alerts:
    # 1. High AI risk score on the current view date.
    # 2. Recently tested positive (e.g., within last 3 days of view date).
    # 3. Pending referrals.
    # 4. TB cases not yet contact traced.
    latest_view_date = df_chw_view['date'].max()
    recent_test_cutoff = latest_view_date - pd.Timedelta(days=3)

    alert_conditions = (
        (df_chw_view['ai_risk_score'] >= risk_thresh) |
        ((df_chw_view['test_result'] == 'Positive') & (df_chw_view['test_date'] >= recent_test_cutoff)) |
        (df_chw_view['referral_status'] == 'Pending') |
        ((df_chw_view['condition'] == 'TB') & (df_chw_view['tb_contact_traced'] == 0))
    )
    alerts_df = df_chw_view[alert_conditions].copy()

    if alerts_df.empty: return pd.DataFrame()

    def determine_reason(row):
        if row['ai_risk_score'] >= risk_thresh: return f"High Risk ({row['ai_risk_score']:.0f})"
        if row['test_result'] == 'Positive' and row['test_date'] >= recent_test_cutoff: return f"Positive Test ({row['condition']})"
        if row['referral_status'] == 'Pending': return f"Pending Referral ({row['condition']})"
        if row['condition'] == 'TB' and row['tb_contact_traced'] == 0: return "TB Contact Tracing"
        return "Other" # Fallback, should ideally not be reached if logic is tight
        
    alerts_df['alert_reason'] = alerts_df.apply(determine_reason, axis=1)
    
    # Prioritize and deduplicate by patient_id, showing the most critical reason
    alerts_df = alerts_df.sort_values(by=['patient_id', 'ai_risk_score'], ascending=[True, False])
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    
    return alerts_df[['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'date']]


@st.cache_data
def get_clinic_summary(df_clinic_view):
    if df_clinic_view.empty:
        return {"avg_test_turnaround": 0, "positive_test_rate": 0, "critical_supply_items": 0, "pending_tests_count": 0}

    # Test Turnaround: only for rows where it's a valid number
    valid_turnaround_df = df_clinic_view.dropna(subset=['test_turnaround_days'])
    avg_turnaround = valid_turnaround_df['test_turnaround_days'].mean() if not valid_turnaround_df.empty else 0
    
    # Positive Test Rate: consider only tests with a definitive result (not 'Pending' or 'N/A' or 'Unknown')
    conclusive_tests_df = df_clinic_view[~df_clinic_view['test_result'].isin(['Pending', 'N/A', 'Unknown'])]
    positive_tests = conclusive_tests_df[conclusive_tests_df['test_result'] == 'Positive'].shape[0]
    total_conclusive_tests = conclusive_tests_df.shape[0]
    positive_rate = (positive_tests / total_conclusive_tests) * 100 if total_conclusive_tests > 0 else 0
    
    critical_supply = df_clinic_view[df_clinic_view['supply_level_days'] <= app_config.CRITICAL_SUPPLY_DAYS]['supply_item'].nunique()
    pending_tests_count = df_clinic_view[df_clinic_view['test_result'] == 'Pending'].shape[0]

    return {
        "avg_test_turnaround": avg_turnaround, "positive_test_rate": positive_rate,
        "critical_supply_items": critical_supply, "pending_tests_count": pending_tests_count
    }

@st.cache_data
def get_patient_alerts_for_clinic(df_clinic_view, risk_threshold=None):
    if df_clinic_view.empty: return pd.DataFrame()
    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['moderate']

    latest_view_date = df_clinic_view['date'].max() # Assuming df_clinic_view is already date-filtered
    recent_test_cutoff = latest_view_date - pd.Timedelta(days=7) # Look back 7 days for positive tests

    alert_conditions = (
        (df_clinic_view['ai_risk_score'] >= risk_thresh) |
        ((df_clinic_view['test_result'] == 'Positive') & (df_clinic_view['test_date'] >= recent_test_cutoff)) |
        ((df_clinic_view['test_result'] == 'Pending') & (df_clinic_view['condition'].isin(['TB','Malaria']))) # Pending critical tests
    )
    alerts_df = df_clinic_view[alert_conditions].copy()

    if alerts_df.empty: return pd.DataFrame()

    def determine_reason_clinic(row):
        if row['ai_risk_score'] >= risk_thresh: return f"High Risk ({row['ai_risk_score']:.0f})"
        if row['test_result'] == 'Positive' and row['test_date'] >= recent_test_cutoff: return f"Recent Positive Test ({row['condition']})"
        if row['test_result'] == 'Pending' and row['condition'].isin(['TB','Malaria']): return f"Pending Critical Test ({row['condition']})"
        return "Review Case"
        
    alerts_df['alert_reason'] = alerts_df.apply(determine_reason_clinic, axis=1)
    alerts_df = alerts_df.sort_values(by=['patient_id', 'ai_risk_score'], ascending=[True, False])
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    
    return alerts_df[['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result', 'referral_status', 'alert_reason', 'date']]


@st.cache_data
def get_trend_data(df, value_col, date_col='date', period='D', agg_func='mean'):
    if df.empty or value_col not in df.columns or date_col not in df.columns:
        logger.warning(f"Cannot generate trend for '{value_col}': DataFrame empty or columns missing.")
        return pd.Series(dtype='float64').rename_axis(date_col if date_col in df.columns else 'date')
    
    # Ensure date_col is datetime and handle potential errors
    try:
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy.dropna(subset=[date_col, value_col], inplace=True) # Drop rows where date or value is NaN
        
        if df_copy.empty:
             logger.warning(f"No valid data points left for trend '{value_col}' after NaT/NaN drop.")
             return pd.Series(dtype='float64').rename_axis(date_col)

        if agg_func == 'count': # For counts, we count occurrences per period
            trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].count()
        elif agg_func == 'sum':
            trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].sum()
        else: # Default to mean
            trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].mean()
        
        return trend.fillna(0) # Fill any resulting NaNs from Grouper (e.g., empty periods)
    except Exception as e:
        logger.error(f"Error generating trend data for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype='float64').rename_axis(date_col if date_col in df.columns else 'date')

@st.cache_data
def get_supply_forecast_data(df_health_records, forecast_days_out=7):
    if df_health_records.empty: return pd.DataFrame()

    # Get the latest supply level for each item
    latest_supplies = df_health_records.sort_values('date').drop_duplicates(subset=['supply_item'], keep='last')
    latest_supplies = latest_supplies[latest_supplies['supply_level_days'].notna() & (latest_supplies['supply_level_days'] > 0)]

    if latest_supplies.empty: return pd.DataFrame()

    forecast_list = []
    for _, row in latest_supplies.iterrows():
        item = row['supply_item']
        current_level_days = row['supply_level_days']
        last_recorded_date = row['date']
        
        # Simple linear depletion model: assume current_level_days will deplete in that many days
        # Daily consumption rate = 1 (since level is in days)
        # More advanced: calculate historical consumption rate if data allows
        daily_consumption = 1.0 

        for i in range(1, forecast_days_out + 1):
            forecast_date = last_recorded_date + pd.Timedelta(days=i)
            forecasted_level = max(0, current_level_days - (daily_consumption * i))
            
            # Simple CI: +/- 10-20% of forecasted level, or fixed bands
            ci_factor = 0.15 
            lower_ci = max(0, forecasted_level * (1 - ci_factor))
            upper_ci = forecasted_level * (1 + ci_factor)
            
            forecast_list.append({
                'date': forecast_date,
                'item': item,
                'forecast_days': forecasted_level,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            })
    
    return pd.DataFrame(forecast_list) if forecast_list else pd.DataFrame()

@st.cache_data
def get_district_summary_kpis(enriched_zone_gdf):
    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
        return {"avg_population_risk": 0, "overall_facility_coverage": 0, "zones_high_risk": 0}

    # Ensure 'population' column exists and is numeric for weighting
    if 'population' not in enriched_zone_gdf.columns or not pd.api.types.is_numeric_dtype(enriched_zone_gdf['population']):
        logger.warning("Population data missing or non-numeric in enriched GDF. Cannot calculate weighted KPIs.")
        return {"avg_population_risk": enriched_zone_gdf['avg_risk_score'].mean() if 'avg_risk_score' in enriched_zone_gdf else 0, 
                "overall_facility_coverage": enriched_zone_gdf['facility_coverage_score'].mean() if 'facility_coverage_score' in enriched_zone_gdf else 0, 
                "zones_high_risk": enriched_zone_gdf[enriched_zone_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0] if 'avg_risk_score' in enriched_zone_gdf else 0}

    total_population = enriched_zone_gdf['population'].sum()
    
    avg_pop_risk = 0
    if 'avg_risk_score' in enriched_zone_gdf.columns and total_population > 0:
        avg_pop_risk = (enriched_zone_gdf['avg_risk_score'] * enriched_zone_gdf['population']).sum() / total_population
    
    overall_facility_coverage = 0
    if 'facility_coverage_score' in enriched_zone_gdf.columns and total_population > 0:
        overall_facility_coverage = (enriched_zone_gdf['facility_coverage_score'] * enriched_zone_gdf['population']).sum() / total_population
                                
    zones_high_risk = 0
    if 'avg_risk_score' in enriched_zone_gdf.columns:
        zones_high_risk = enriched_zone_gdf[enriched_zone_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]

    return {
        "avg_population_risk": avg_pop_risk,
        "overall_facility_coverage": overall_facility_coverage,
        "zones_high_risk": zones_high_risk
    }
