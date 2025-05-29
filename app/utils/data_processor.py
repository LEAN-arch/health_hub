import pandas as pd
import geopandas as gpd
import os
import logging
import streamlit as st # For caching

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # health_hub directory
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_data(ttl=3600) # Cache for 1 hour
def load_health_data(file_name="sample_health_data.csv"):
    """
    Load and validate main health CSV data.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            st.error(f"Data file not found: {file_path}. Please ensure it exists in the 'data' directory.")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path, parse_dates=['date', 'referral_date', 'test_date'])
        
        required_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns in health data: {missing}")
            st.error(f"Health data is missing required columns: {missing}. Please check the CSV file.")
            return pd.DataFrame()
        
        # Basic type conversions and cleaning
        numeric_cols = ['age', 'chw_visit', 'test_turnaround_days', 'supply_level_days', 'ai_risk_score', 'tb_contact_traced']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        logger.info(f"Successfully loaded and validated health data from {file_path} with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading health data from {file_path}: {str(e)}")
        st.error(f"An error occurred while loading health data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_geojson_data(file_name="zones.geojson"):
    """
    Load and validate GeoJSON data for zones.
    Returns GeoDataFrame.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        if not os.path.exists(file_path):
            logger.error(f"GeoJSON file not found: {file_path}")
            st.error(f"GeoJSON file not found: {file_path}. Please ensure it exists in the 'data' directory.")
            return None # Or an empty GeoDataFrame: gpd.GeoDataFrame()
        
        gdf = gpd.read_file(file_path)
        
        required_props = ["zone_id", "name", "population"] # From our GeoJSON definition
        if not all(prop in gdf.columns for prop in required_props):
            missing = [prop for prop in required_props if prop not in gdf.columns]
            logger.error(f"Missing required properties in GeoJSON: {missing}")
            st.error(f"GeoJSON is missing required properties: {missing}. Please check the GeoJSON file.")
            return None # Or an empty GeoDataFrame
        
        logger.info(f"Successfully loaded GeoJSON data from {file_path} with {len(gdf)} features.")
        return gdf
    except Exception as e:
        logger.error(f"Error loading GeoJSON data from {file_path}: {str(e)}")
        st.error(f"An error occurred while loading GeoJSON data: {str(e)}")
        return None # Or an empty GeoDataFrame

def get_overall_kpis(df):
    """Calculates high-level KPIs from the main health dataframe."""
    if df.empty:
        return {
            "total_patients": 0, "active_cases": 0, "avg_risk_score": 0,
            "pending_referrals": 0, "positive_tests_past_week": 0
        }
    
    latest_date = df['date'].max()
    one_week_ago = latest_date - pd.Timedelta(days=7)

    total_patients = df['patient_id'].nunique()
    # Define active cases (example: specific conditions or recent positive tests)
    active_cases = df[df['condition'].isin(['TB', 'Malaria'])]['patient_id'].nunique() # Simplified
    avg_risk_score = df['ai_risk_score'].mean()
    pending_referrals = df[df['referral_status'] == 'Pending']['patient_id'].nunique()
    
    positive_tests = df[
        (df['test_result'] == 'Positive') & (df['test_date'] >= one_week_ago)
    ]['patient_id'].nunique()

    return {
        "total_patients": total_patients,
        "active_cases": active_cases,
        "avg_risk_score": avg_risk_score,
        "pending_referrals": pending_referrals,
        "positive_tests_past_week": positive_tests
    }

def get_chw_summary(df, chw_id=None): # chw_id for future use if CHWs are assigned patients
    """Calculates summary for CHW dashboard."""
    if df.empty:
        return {"visits_today": 0, "pending_tasks": 0, "high_risk_followups": 0}

    today = df['date'].max() # Assuming latest date in data is 'today' for mock
    visits_today = df[(df['date'] == today) & (df['chw_visit'] == 1)].shape[0]
    
    # Pending tasks: referrals + TB contacts needing tracing
    pending_referrals = df[df['referral_status'] == 'Pending'].shape[0]
    pending_tb_trace = df[(df['condition'] == 'TB') & (df['tb_contact_traced'] == 0)].shape[0]
    pending_tasks = pending_referrals + pending_tb_trace
    
    high_risk_followups = df[(df['ai_risk_score'] >= 75) & (df['referral_status'] != 'Completed')].shape[0]
    
    return {
        "visits_today": visits_today,
        "pending_tasks": pending_tasks,
        "high_risk_followups": high_risk_followups
    }

def get_patient_alerts(df, risk_threshold=75, recent_days=7):
    """Identifies patients needing urgent attention."""
    if df.empty:
        return pd.DataFrame()
    
    latest_date = df['date'].max()
    recent_cutoff = latest_date - pd.Timedelta(days=recent_days)
    
    alerts = df[
        ((df['ai_risk_score'] >= risk_threshold) & (df['date'] >= recent_cutoff)) |
        (df['test_result'] == 'Positive') & (df['test_date'] >= recent_cutoff) |
        (df['referral_status'] == 'Pending') # All pending referrals are alerts
    ].copy()
    
    alerts['alert_reason'] = 'High Risk Score' # Default
    alerts.loc[alerts['test_result'] == 'Positive', 'alert_reason'] = 'Positive Test (' + alerts['condition'] + ')'
    alerts.loc[alerts['referral_status'] == 'Pending', 'alert_reason'] = 'Pending Referral (' + alerts['condition'] + ')'
    
    alerts = alerts.sort_values(by='ai_risk_score', ascending=False)
    return alerts[['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'date']].drop_duplicates(subset=['patient_id'])


def get_trend_data(df, value_col, date_col='date', period='D'):
    """Generates time series trend data."""
    if df.empty or value_col not in df.columns or date_col not in df.columns:
        return pd.Series(dtype='float64').rename_axis(date_col)
    
    # Ensure date_col is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    trend = df.groupby(pd.Grouper(key=date_col, freq=period))[value_col].mean() # or .sum() or .count()
    return trend.fillna(0)


def get_clinic_summary(df):
    if df.empty:
        return {
            "avg_test_turnaround": 0, "positive_test_rate": 0,
            "critical_supply_items": 0, "pending_tests_count": 0
        }
    
    avg_turnaround = df['test_turnaround_days'].mean()
    
    tested_df = df.dropna(subset=['test_result'])
    positive_tests = tested_df[tested_df['test_result'] == 'Positive'].shape[0]
    total_tests = tested_df.shape[0]
    positive_rate = (positive_tests / total_tests) * 100 if total_tests > 0 else 0
    
    critical_supply = df[df['supply_level_days'] <= 7]['supply_item'].nunique() # 7 days as critical
    pending_tests_count = df[df['test_result'] == 'Pending'].shape[0]

    return {
        "avg_test_turnaround": avg_turnaround,
        "positive_test_rate": positive_rate,
        "critical_supply_items": critical_supply,
        "pending_tests_count": pending_tests_count
    }

def get_supply_forecast_data(df):
    """Generates mock forecast data for supplies."""
    if df.empty:
        return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])

    # Simplified: take last known supply level and linearly decrease, add some noise for CI
    # In reality, this would use a proper forecasting model
    forecast_data = []
    latest_supplies = df.sort_values('date').drop_duplicates(subset=['supply_item'], keep='last')
    
    for _, row in latest_supplies.iterrows():
        if pd.isna(row['supply_level_days']): continue
        current_level = row['supply_level_days']
        for i in range(1, 8): # Forecast for next 7 days
            forecast_date = row['date'] + pd.Timedelta(days=i)
            # Simple linear decrease, ensure non-negative
            forecast_val = max(0, current_level - (i * (current_level / 30 if current_level > 0 else 1) ) ) # Assuming avg consumption depletes in 30 days
            
            forecast_data.append({
                'date': forecast_date,
                'item': row['supply_item'],
                'forecast_days': forecast_val,
                'lower_ci': max(0, forecast_val * 0.9), # 10% CI
                'upper_ci': forecast_val * 1.1
            })
    return pd.DataFrame(forecast_data)


def merge_health_data_with_geojson(health_df, geojson_gdf):
    """Merges aggregated health data with GeoJSON data for mapping."""
    if health_df.empty or geojson_gdf is None or geojson_gdf.empty:
        return geojson_gdf # Return original geojson if no data to merge

    # Aggregate health data by zone_id
    # Example: average risk score, count of active cases per zone
    zone_summary = health_df.groupby('zone_id').agg(
        avg_risk_score=('ai_risk_score', 'mean'),
        active_cases=('patient_id', lambda x: health_df.loc[x.index, 'condition'].isin(['TB', 'Malaria']).sum()), # Example for active cases
        total_population_health_data=('patient_id','nunique') # A proxy if actual population per zone isn't always used from geojson
    ).reset_index()
    
    # Merge with GeoDataFrame
    # Ensure 'zone_id' exists in both and has compatible types
    if 'zone_id' not in geojson_gdf.columns:
        logger.error("GeoJSON is missing 'zone_id' property for merging.")
        return geojson_gdf
        
    merged_gdf = geojson_gdf.merge(zone_summary, on='zone_id', how='left')
    merged_gdf['avg_risk_score'] = merged_gdf['avg_risk_score'].fillna(0)
    merged_gdf['active_cases'] = merged_gdf['active_cases'].fillna(0)
    
    # Calculate prevalence per 1000 using GeoJSON population
    merged_gdf['prevalence_per_1000'] = \
        (merged_gdf['active_cases'] / merged_gdf['population']) * 1000 \
        if 'population' in merged_gdf.columns and merged_gdf['population'].notna().all() and (merged_gdf['population'] > 0).all() \
        else 0
    
    # Example: Calculate facility coverage score
    # Lower travel time and more clinics = better coverage. Normalize to 0-100.
    if 'avg_travel_time_clinic_min' in merged_gdf.columns and 'num_clinics' in merged_gdf.columns:
        # Normalize travel time (lower is better, so invert)
        max_travel_time = merged_gdf['avg_travel_time_clinic_min'].max()
        min_travel_time = merged_gdf['avg_travel_time_clinic_min'].min()
        if max_travel_time == min_travel_time : # Avoid division by zero if all values are same
            merged_gdf['travel_score'] = 50 # Assign a neutral score
        else:
            merged_gdf['travel_score'] = 100 * (1 - (merged_gdf['avg_travel_time_clinic_min'] - min_travel_time) / (max_travel_time - min_travel_time))


        # Normalize num_clinics (higher is better)
        max_clinics = merged_gdf['num_clinics'].max()
        min_clinics = merged_gdf['num_clinics'].min()
        if max_clinics == min_clinics:
            merged_gdf['clinic_count_score'] = 50
        else:
            merged_gdf['clinic_count_score'] = 100 * (merged_gdf['num_clinics'] - min_clinics) / (max_clinics - min_clinics)
        
        merged_gdf['facility_coverage_score'] = (merged_gdf['travel_score'] * 0.6 + merged_gdf['clinic_count_score'] * 0.4).fillna(50)
    else:
        merged_gdf['facility_coverage_score'] = 50 # Default if data missing

    return merged_gdf

def get_district_summary_kpis(merged_gdf):
    """Calculates KPIs for district level from merged GeoDataFrame."""
    if merged_gdf is None or merged_gdf.empty:
        return {
            "avg_population_risk": 0,
            "overall_facility_coverage": 0,
            "zones_high_risk": 0
        }

    avg_pop_risk = (merged_gdf['avg_risk_score'] * merged_gdf['population']).sum() / merged_gdf['population'].sum() \
                   if 'population' in merged_gdf.columns and merged_gdf['population'].sum() > 0 else 0
    
    overall_facility_coverage = (merged_gdf['facility_coverage_score'] * merged_gdf['population']).sum() / merged_gdf['population'].sum() \
                                if 'population' in merged_gdf.columns and merged_gdf['population'].sum() > 0 else 0
                                
    zones_high_risk = merged_gdf[merged_gdf['avg_risk_score'] >= 70].shape[0] # Zones with avg risk score >= 70

    return {
        "avg_population_risk": avg_pop_risk,
        "overall_facility_coverage": overall_facility_coverage,
        "zones_high_risk": zones_high_risk
    }
