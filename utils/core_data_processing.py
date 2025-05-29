# utils/core_data_processing.py
import pandas as pd
import geopandas as gpd
import os
import logging
import streamlit as st
import numpy as np # Ensure numpy is imported for np.nan
from config import app_config

logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def hash_geodataframe(gdf): # pragma: no cover
    if gdf is None: return None
    if not isinstance(gdf, gpd.GeoDataFrame):
        if isinstance(gdf, pd.DataFrame):
            try: return gdf.to_parquet()
            except Exception: return str(gdf)
        return gdf
    try:
        geom_col_name = gdf.geometry.name
        geometry_hash_part = b""
        if geom_col_name in gdf.columns and not gdf[geom_col_name].empty:
            valid_geoms = gdf[geom_col_name][gdf[geom_col_name].is_valid & ~gdf[geom_col_name].is_empty]
            if not valid_geoms.empty: geometry_hash_part = valid_geoms.to_wkt().values.tobytes()
        data_df = gdf.drop(columns=[geom_col_name], errors='ignore')
        data_hash_part = b""
        if not data_df.empty:
            data_df = data_df.reindex(sorted(data_df.columns), axis=1)
            data_hash_part = data_df.to_parquet()
        return (data_hash_part, geometry_hash_part)
    except Exception as e:
        logger.error(f"Error hashing GeoDataFrame (falling back to string): {e}", exc_info=True)
        return str(gdf)


@st.cache_data(ttl=3600)
def load_health_records():
    file_path = app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    try:
        if not os.path.exists(file_path): # pragma: no cover
            logger.error(f"Health records file not found: {file_path}")
            st.error(f"Data file '{os.path.basename(file_path)}' not found. Check 'data_sources/'.")
            return pd.DataFrame()

        df = pd.read_csv(file_path, parse_dates=['date', 'referral_date', 'test_date'])

        required_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score"]
        if not all(col in df.columns for col in required_cols): # pragma: no cover
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns in health records: {missing}")
            st.error(f"Health records missing required columns: {missing}. Check CSV.")
            return pd.DataFrame()

        # Define all expected numeric columns, including new sensor data
        numeric_cols_expected = [
            'age', 'chw_visit', 'test_turnaround_days', 'supply_level_days',
            'ai_risk_score', 'tb_contact_traced', 'avg_daily_steps',
            'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs',
            'sleep_score_pct', 'stress_level_score', 'avg_skin_temp_celsius',
            'max_skin_temp_celsius', 'avg_spo2_pct', 'min_spo2_pct', 'fall_detected_today'
        ]
        for col in numeric_cols_expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.debug(f"Numeric column '{col}' not found in health_records.csv. Creating with NaNs.")
                df[col] = np.nan # Initialize missing numeric columns with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure correct dtype

        string_cols_expected = [
            'referral_status', 'test_result', 'test_type', 'supply_item',
            'gender', 'condition', 'zone_id', 'patient_id'
        ]
        for col in string_cols_expected:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
            else:
                logger.debug(f"String column '{col}' not found in health_records.csv. Creating with 'Unknown'.")
                df[col] = 'Unknown'
                df[col] = df[col].astype(str) # Ensure correct dtype

        logger.info(f"Loaded health records from {file_path} ({len(df)} rows).")
        return df
    except pd.errors.EmptyDataError: # pragma: no cover
        logger.error(f"Health records file is empty: {file_path}")
        st.error(f"Health records file '{os.path.basename(file_path)}' is empty.")
        return pd.DataFrame()
    except Exception as e: # pragma: no cover
        logger.error(f"Error loading health records from {file_path}: {e}", exc_info=True)
        st.error(f"Unexpected error loading health records: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def load_zone_data():
    # ... (Content as in the previous "complete, functional code" response - this function was stable)
    attributes_path = app_config.ZONE_ATTRIBUTES_CSV
    geometries_path = app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"Loading zone attributes: {attributes_path}, geometries: {geometries_path}")
    try:
        if not os.path.exists(attributes_path): # pragma: no cover
            logger.error(f"Zone attributes file not found: {attributes_path}")
            st.error(f"Zone attributes file '{os.path.basename(attributes_path)}' not found.")
            return None
        zone_attributes_df = pd.read_csv(attributes_path)
        required_attr = ["zone_id", "name", "population", "socio_economic_index", "num_clinics", "avg_travel_time_clinic_min"]
        if not all(col in zone_attributes_df.columns for col in required_attr): # pragma: no cover
            missing = [c for c in required_attr if c not in zone_attributes_df.columns]
            logger.error(f"Zone attributes CSV missing: {missing}")
            st.error(f"Zone attributes CSV missing: {missing}.")
            return None

        if not os.path.exists(geometries_path): # pragma: no cover
            logger.error(f"Zone geometries file not found: {geometries_path}")
            st.error(f"Zone geometries file '{os.path.basename(geometries_path)}' not found.")
            return None
        zone_geometries_gdf = gpd.read_file(geometries_path)
        if "zone_id" not in zone_geometries_gdf.columns: # pragma: no cover
            logger.error("Zone geometries GeoJSON missing 'zone_id' property.")
            st.error("Zone geometries GeoJSON missing 'zone_id'.")
            return None

        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str)
        zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str)
        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left")
        if merged_gdf['name'].isnull().any(): # pragma: no cover
            logger.warning(f"Unmatched zone geometries/attributes: {merged_gdf[merged_gdf['name'].isnull()]['zone_id'].tolist()}")
        logger.info(f"Loaded and merged zone data ({len(merged_gdf)} features).")
        return merged_gdf
    except Exception as e: # pragma: no cover
        logger.error(f"Error loading zone data: {e}", exc_info=True)
        st.error(f"Error loading zone data: {e}")
        return None


@st.cache_data(ttl=3600)
def load_iot_clinic_environment_data():
    # ... (Content as in the previous "complete, functional code" response - this function was stable)
    file_path = app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"Attempting to load IoT clinic environment data from: {file_path}")
    try:
        if not os.path.exists(file_path): # pragma: no cover
            logger.warning(f"IoT data file '{os.path.basename(file_path)}' not found. Environment metrics unavailable.")
            return pd.DataFrame()

        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        required_cols = ['timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm'] # Min example
        if not all(col in df.columns for col in required_cols): # pragma: no cover
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"IoT data missing required columns: {missing}. Environment metrics may be incomplete.")
            if 'timestamp' not in df.columns or 'clinic_id' not in df.columns: return pd.DataFrame()

        if 'clinic_id' in df.columns:
            try:
                df['zone_id'] = df['clinic_id'].apply(lambda x: "-".join(str(x).split('-')[:2]) if isinstance(x, str) and x.count('-') >= 1 else 'UnknownZone')
            except Exception: df['zone_id'] = 'UnknownZone' # pragma: no cover
        
        numeric_iot_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index',
                            'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
                            'waiting_room_occupancy', 'patient_throughput_per_hour',
                            'sanitizer_dispenses_per_hour']
        for col in numeric_iot_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: # pragma: no cover
                 logger.debug(f"Numeric IoT column '{col}' not found in iot_clinic_environment.csv.")
                 df[col] = np.nan # Initialize with NaN
                 df[col] = pd.to_numeric(df[col], errors='coerce')


        logger.info(f"Loaded IoT clinic environment data ({len(df)} rows).")
        return df
    except pd.errors.EmptyDataError: # pragma: no cover
        logger.warning(f"IoT data file '{os.path.basename(file_path)}' is empty.")
        return pd.DataFrame()
    except Exception as e: # pragma: no cover
        logger.error(f"Error loading IoT clinic environment data: {e}", exc_info=True)
        st.error(f"Error loading IoT clinic environment data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, hash_funcs={gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def enrich_zone_geodata_with_health_aggregates(zone_gdf, health_df, iot_df=None):
    # ... (This function should be mostly the same as the previous "complete, functional code" version)
    # Key is to ensure it gracefully handles potentially missing new sensor columns in health_df
    # by using .get() with defaults or checking for column existence before aggregation.
    base_cols_to_ensure = ['avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases',
                           'pneumonia_cases', 'anemia_cases', 'sti_cases', 'ntd_chagas_cases', 'hpv_screenings_done',
                           'avg_daily_steps_zone', 'avg_sleep_score_zone', 'avg_spo2_zone', 'avg_skin_temp_zone', 'total_falls_detected_zone',
                           'prevalence_per_1000', 'facility_coverage_score', 'total_tests_conducted', 'chw_visits_in_zone',
                           'zone_avg_co2', 'zone_avg_temp', 'total_active_key_infections']

    if zone_gdf is None or zone_gdf.empty:
        logger.warning("Zone GeoDataFrame is empty or None for enrichment. Returning empty GDF with expected schema.")
        return gpd.GeoDataFrame({col: [] for col in (zone_gdf.columns if zone_gdf is not None and not zone_gdf.empty else []) + base_cols_to_ensure}, 
                                crs=(zone_gdf.crs if zone_gdf is not None and not zone_gdf.empty else "EPSG:4326"))

    enriched_gdf = zone_gdf.copy()
    enriched_gdf['zone_id'] = enriched_gdf['zone_id'].astype(str)

    # Initialize all expected aggregate columns to 0.0 or handle specific types
    for col in base_cols_to_ensure:
        if col not in enriched_gdf.columns: enriched_gdf[col] = 0.0

    if health_df is not None and not health_df.empty:
        health_df_copy = health_df.copy()
        health_df_copy['zone_id'] = health_df_copy['zone_id'].astype(str)

        def count_unique_patients_with_condition(condition_list_str_or_list):
            if 'condition' not in health_df_copy.columns: return lambda x: 0
            actual_condition_list = [condition_list_str_or_list] if not isinstance(condition_list_str_or_list, list) else condition_list_str_or_list
            return lambda x: health_df_copy.loc[x.index]['condition'].astype(str).isin(actual_condition_list).nunique() if not x.empty else 0
        
        def count_sti_cases(x_indices):
            if 'condition' not in health_df_copy.columns or x_indices.empty: return 0
            return health_df_copy.loc[x_indices]['condition'].astype(str).str.startswith('STI-', na=False).nunique() # Unique patients with any STI

        zone_summary_aggregations = {
            'avg_risk_score': ('ai_risk_score', 'mean'), 'chw_visits_in_zone': ('chw_visit', 'sum'),
            'active_tb_cases': ('patient_id', count_unique_patients_with_condition(['TB'])),
            'active_malaria_cases': ('patient_id', count_unique_patients_with_condition(['Malaria'])),
            'hiv_positive_cases': ('patient_id', count_unique_patients_with_condition(['HIV-Positive'])),
            'pneumonia_cases': ('patient_id', count_unique_patients_with_condition(['Pneumonia'])),
            'anemia_cases': ('patient_id', count_unique_patients_with_condition(['Anemia'])),
            'sti_cases': ('patient_id', count_sti_cases),
            'ntd_chagas_cases': ('patient_id', count_unique_patients_with_condition(['NTD-Chagas'])),
            'hpv_screenings_done': ('test_type', lambda x: (health_df_copy.loc[x.index]['test_type'].astype(str) == 'PapSmear').sum() if 'test_type' in health_df_copy.columns and not x.empty else 0),
            'avg_daily_steps_zone': ('avg_daily_steps', 'mean'), 'avg_sleep_score_zone': ('sleep_score_pct', 'mean'),
            'avg_spo2_zone': ('avg_spo2_pct', 'mean'), 'avg_skin_temp_zone': ('avg_skin_temp_celsius', 'mean'),
            'total_falls_detected_zone': ('fall_detected_today', 'sum')
        }
        
        valid_aggregations = { k: v for k, v in zone_summary_aggregations.items() if isinstance(v, tuple) and v[0] in health_df_copy.columns }
        if valid_aggregations:
            zone_summary = health_df_copy.groupby('zone_id').agg(**valid_aggregations).reset_index()
            enriched_gdf = enriched_gdf.merge(zone_summary, on='zone_id', how='left')
        else: logger.warning("No valid columns found in health_df for primary aggregation during enrichment.")
    
    # Fill NaNs for all health aggregates (important after merge or if health_df was empty)
    for col in expected_health_agg_cols: # Use the pre-defined list from top of function
        if col not in enriched_gdf.columns: enriched_gdf[col] = 0.0 # Ensure column exists
        # Ensure column is numeric before fillna, especially if it was just added as 0.0
        enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce').fillna(0)


    if iot_df is not None and not iot_df.empty:
        # ... (IoT aggregation and merge as before, ensure robustness) ...
        iot_df_copy = iot_df.copy()
        if 'zone_id' not in iot_df_copy.columns: logger.warning("zone_id missing in IoT data for enrichment.")
        else:
            iot_df_copy['zone_id'] = iot_df_copy['zone_id'].astype(str)
            for iot_col_num in ['avg_co2_ppm', 'max_co2_ppm', 'avg_temp_celsius']:
                if iot_col_num in iot_df_copy.columns: iot_df_copy[iot_col_num] = pd.to_numeric(iot_df_copy[iot_col_num], errors='coerce')
                else: iot_df_copy[iot_col_num] = np.nan
            latest_iot_zone = iot_df_copy.sort_values('timestamp').drop_duplicates(subset=['zone_id', 'room_name'], keep='last')
            iot_zone_summary = latest_iot_zone.groupby('zone_id').agg(zone_avg_co2=('avg_co2_ppm', 'mean'), zone_max_co2=('max_co2_ppm', 'max'), zone_avg_temp=('avg_temp_celsius', 'mean')).reset_index()
            enriched_gdf = enriched_gdf.merge(iot_zone_summary, on='zone_id', how='left')
    
    expected_iot_agg_cols = ['zone_avg_co2', 'zone_max_co2', 'zone_avg_temp']
    for col in expected_iot_agg_cols:
        if col not in enriched_gdf.columns: enriched_gdf[col] = 0.0
        enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce').fillna(0)

    # --- Recalculate total_active_key_infections and prevalence_per_1000 ---
    key_infectious_cols = ['active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'sti_cases']
    enriched_gdf['total_active_key_infections'] = 0.0
    for col_inf in key_infectious_cols:
        if col_inf in enriched_gdf.columns: enriched_gdf['total_active_key_infections'] += pd.to_numeric(enriched_gdf[col_inf], errors='coerce').fillna(0)
        else: enriched_gdf[col_inf] = 0.0 # Ensure column exists if it was missed

    if 'population' in enriched_gdf.columns and 'total_active_key_infections' in enriched_gdf.columns:
        enriched_gdf['population'] = pd.to_numeric(enriched_gdf['population'], errors='coerce').fillna(0)
        enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(lambda row: (row['total_active_key_infections'] / row['population']) * 1000 if row['population'] > 0 else 0.0, axis=1)
    else: enriched_gdf['prevalence_per_1000'] = 0.0
    
    # --- Recalculate facility_coverage_score ---
    if 'avg_travel_time_clinic_min' in enriched_gdf.columns and 'num_clinics' in enriched_gdf.columns:
        # ... (facility coverage score logic as before, ensuring robust fillna) ...
        enriched_gdf['avg_travel_time_clinic_min'] = pd.to_numeric(enriched_gdf['avg_travel_time_clinic_min'], errors='coerce')
        enriched_gdf['num_clinics'] = pd.to_numeric(enriched_gdf['num_clinics'], errors='coerce')
        min_travel = enriched_gdf['avg_travel_time_clinic_min'].min(); max_travel = enriched_gdf['avg_travel_time_clinic_min'].max()
        if pd.isna(min_travel) or pd.isna(max_travel) or max_travel == min_travel: enriched_gdf['travel_score'] = 50.0
        else: enriched_gdf['travel_score'] = 100 * (1 - (enriched_gdf['avg_travel_time_clinic_min'].fillna(max_travel) - min_travel) / (max_travel - min_travel))
        min_clinics = enriched_gdf['num_clinics'].min(); max_clinics = enriched_gdf['num_clinics'].max()
        if pd.isna(min_clinics) or pd.isna(max_clinics) or max_clinics == min_clinics: enriched_gdf['clinic_count_score'] = 50.0
        else: enriched_gdf['clinic_count_score'] = 100 * (enriched_gdf['num_clinics'].fillna(min_clinics) - min_clinics) / (max_clinics - min_clinics)
        enriched_gdf['facility_coverage_score'] = (enriched_gdf['travel_score'].fillna(50) * 0.6 + enriched_gdf['clinic_count_score'].fillna(50) * 0.4)
    else: enriched_gdf['facility_coverage_score'] = 50.0

    for score_col_final in ['travel_score', 'clinic_count_score', 'facility_coverage_score', 'prevalence_per_1000']:
        if score_col_final not in enriched_gdf.columns: enriched_gdf[score_col_final] = 0.0
        enriched_gdf[score_col_final] = pd.to_numeric(enriched_gdf[score_col_final], errors='coerce').fillna(
            enriched_gdf[score_col_final].median() if pd.api.types.is_numeric_dtype(enriched_gdf[score_col_final]) and not enriched_gdf[score_col_final].empty and enriched_gdf[score_col_final].notna().any() else 0.0
        )
    logger.info("Zone geodata successfully enriched.")
    return enriched_gdf


# --- KPI Calculation Functions ---
@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_overall_kpis(df_health_records):
    if df_health_records.empty: return {"total_patients": 0, "tb_cases_active": 0, "malaria_rdt_positive_rate":0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals": 0}
    df = df_health_records.copy() # Operate on a copy
    if 'date' not in df.columns or df['date'].isnull().all(): return {"total_patients": 0, "tb_cases_active": 0, "malaria_rdt_positive_rate":0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals": 0}
    df['date'] = pd.to_datetime(df['date'], errors='coerce'); df.dropna(subset=['date'], inplace=True)
    if df.empty: return {"total_patients": 0, "tb_cases_active": 0, "malaria_rdt_positive_rate":0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals": 0}
    
    latest_date = df['date'].max(); period_start_date = latest_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND)
    
    # Ensure columns used for KPIs exist and have appropriate types/fillings
    df['patient_id'] = df.get('patient_id', pd.Series(dtype='str')).astype(str).fillna('Unknown')
    df['condition'] = df.get('condition', pd.Series(dtype='str')).astype(str).fillna('Unknown')
    df['test_result'] = df.get('test_result', pd.Series(dtype='str')).astype(str).fillna('Unknown')
    df['referral_status'] = df.get('referral_status', pd.Series(dtype='str')).astype(str).fillna('Unknown')
    df['test_type'] = df.get('test_type', pd.Series(dtype='str')).astype(str).fillna('Unknown')
    df['test_date'] = pd.to_datetime(df.get('test_date'), errors='coerce')


    total_patients = df['patient_id'].nunique()
    tb_cases_active = df[(df['condition'] == 'TB') & (df['test_result'] == 'Positive') & (df['referral_status'] != 'Completed')]['patient_id'].nunique()
    
    malaria_rdt_tests = df[df['test_type'] == 'RDT-Malaria']
    malaria_rdt_positive = malaria_rdt_tests[malaria_rdt_tests['test_result'] == 'Positive'].shape[0]
    malaria_rdt_total_conclusive = malaria_rdt_tests[~malaria_rdt_tests['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])].shape[0]
    malaria_rdt_positive_rate = (malaria_rdt_positive / malaria_rdt_total_conclusive) * 100 if malaria_rdt_total_conclusive > 0 else 0.0
    
    hiv_newly_diagnosed = df[
        (df['condition'] == 'HIV-Positive') & 
        (df['test_result'] == 'Positive') & 
        (df['test_type'].str.contains("HIV", case=False, na=False)) & 
        (df['test_date'].notna()) & (df['test_date'] >= period_start_date)
    ]['patient_id'].nunique()
    
    pending_critical_referrals = df[(df['referral_status'] == 'Pending') & (df['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS))]['patient_id'].nunique()
    
    return {"total_patients": total_patients, "tb_cases_active": tb_cases_active, "malaria_rdt_positive_rate": malaria_rdt_positive_rate, "hiv_newly_diagnosed_period": hiv_newly_diagnosed, "pending_critical_referrals": pending_critical_referrals}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_chw_summary(df_chw_view):
    if df_chw_view.empty: return {"visits_today": 0, "tb_contacts_to_trace": 0, "sti_symptomatic_referrals": 0, "avg_patient_risk_visited":0.0, "patients_low_spo2_visited": 0, "patients_fever_visited": 0}
    
    df = df_chw_view.copy()
    # Ensure all used columns exist and have correct types, with defaults
    df['chw_visit'] = pd.to_numeric(df.get('chw_visit'), errors='coerce').fillna(0)
    df['condition'] = df.get('condition', 'Unknown').astype(str)
    df['tb_contact_traced'] = pd.to_numeric(df.get('tb_contact_traced'), errors='coerce').fillna(0)
    df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score'), errors='coerce') # Keep NaNs for mean
    df['referral_status'] = df.get('referral_status', 'Unknown').astype(str)
    df['avg_spo2_pct'] = pd.to_numeric(df.get('avg_spo2_pct'), errors='coerce') # Keep NaNs
    df['max_skin_temp_celsius'] = pd.to_numeric(df.get('max_skin_temp_celsius'), errors='coerce') # Keep NaNs

    visits_today = df[df['chw_visit'] == 1].shape[0]
    tb_contacts_to_trace = df[(df['condition'] == 'TB') & (df['tb_contact_traced'] == 0)].shape[0]
    sti_symptomatic_referrals = df[(df['condition'].str.startswith("STI-", na=False)) & (df['referral_status'] == 'Pending')].shape[0]
    
    patients_visited_today_df = df[df['chw_visit'] == 1]
    avg_patient_risk_visited = patients_visited_today_df['ai_risk_score'].mean() if not patients_visited_today_df.empty and patients_visited_today_df['ai_risk_score'].notna().any() else 0.0
    
    patients_low_spo2_visited = patients_visited_today_df[patients_visited_today_df['avg_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT].shape[0] if 'avg_spo2_pct' in patients_visited_today_df and patients_visited_today_df['avg_spo2_pct'].notna().any() else 0
    patients_fever_visited = patients_visited_today_df[patients_visited_today_df['max_skin_temp_celsius'] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C].shape[0] if 'max_skin_temp_celsius' in patients_visited_today_df and patients_visited_today_df['max_skin_temp_celsius'].notna().any() else 0

    return {"visits_today": visits_today, "tb_contacts_to_trace": tb_contacts_to_trace, "sti_symptomatic_referrals": sti_symptomatic_referrals, "avg_patient_risk_visited": avg_patient_risk_visited if pd.notna(avg_patient_risk_visited) else 0.0, "patients_low_spo2_visited": patients_low_spo2_visited, "patients_fever_visited": patients_fever_visited}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_chw(df_chw_view, risk_threshold=None):
    if df_chw_view.empty: return pd.DataFrame()
    
    df_copy = df_chw_view.copy()
    df_copy['date'] = pd.to_datetime(df_copy.get('date'), errors='coerce')
    df_copy['test_date'] = pd.to_datetime(df_copy.get('test_date'), errors='coerce')
    df_copy.dropna(subset=['date'], inplace=True)
    if df_copy.empty: return pd.DataFrame()

    # Initialize all potentially used columns robustly
    numeric_cols_to_init_nan = ['ai_risk_score', 'min_spo2_pct', 'max_skin_temp_celsius']
    for col in numeric_cols_to_init_nan:
        if col not in df_copy.columns: df_copy[col] = np.nan
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    numeric_cols_to_init_zero = ['tb_contact_traced', 'fall_detected_today']
    for col in numeric_cols_to_init_zero:
        if col not in df_copy.columns: df_copy[col] = 0
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
    string_cols_to_init_unknown = ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id']
    for col in string_cols_to_init_unknown:
        if col not in df_copy.columns: df_copy[col] = 'Unknown'
        df_copy[col] = df_copy[col].astype(str).fillna('Unknown')

    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['chw_alert_moderate']
    
    # Build conditions safely, ensuring Series for bitwise operations
    cond_risk = df_copy['ai_risk_score'].notna() & (df_copy['ai_risk_score'] >= risk_thresh)
    cond_tb_trace = (df_copy['condition'] == 'TB') & (df_copy['tb_contact_traced'] == 0)
    cond_tb_referral = (df_copy['condition'] == 'TB') & (df_copy['referral_status'] == 'Pending')
    cond_sti_referral = df_copy['condition'].str.startswith("STI-", na=False) & (df_copy['referral_status'] == 'Pending')
    cond_hiv_referral = (df_copy['condition'] == 'HIV-Positive') & (df_copy['referral_status'] == 'Pending')
    cond_pneumonia_risk = (df_copy['condition'] == 'Pneumonia') & df_copy['ai_risk_score'].notna() & (df_copy['ai_risk_score'] >= app_config.RISK_THRESHOLDS['moderate'])
    cond_low_spo2 = df_copy['min_spo2_pct'].notna() & (df_copy['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT)
    cond_fever = df_copy['max_skin_temp_celsius'].notna() & (df_copy['max_skin_temp_celsius'] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C)
    cond_fall = df_copy['fall_detected_today'] > 0

    alert_conditions = cond_risk | cond_tb_trace | cond_tb_referral | cond_sti_referral | \
                       cond_hiv_referral | cond_pneumonia_risk | cond_low_spo2 | cond_fever | cond_fall
    
    alerts_df = df_copy[alert_conditions].copy()
    if alerts_df.empty: return pd.DataFrame()

    def determine_reason_chw_focused_extended(row):
        if pd.notna(row.get('fall_detected_today')) and row.get('fall_detected_today', 0) > 0: return f"Fall Detected"
        if pd.notna(row.get('min_spo2_pct')) and row.get('min_spo2_pct', 101) < app_config.SPO2_LOW_THRESHOLD_PCT: return f"Low SpO2 ({row.get('min_spo2_pct', 0):.0f}%)"
        if pd.notna(row.get('max_skin_temp_celsius')) and row.get('max_skin_temp_celsius', 0) >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C: return f"Fever ({row.get('max_skin_temp_celsius', 0):.1f}°C)"
        if row.get('condition') == 'TB' and row.get('tb_contact_traced', 1) == 0: return "TB Contact Trace"
        if row.get('condition') == 'TB' and row.get('referral_status') == 'Pending': return "TB Referral Pend."
        if str(row.get('condition','')).startswith("STI-") and row.get('referral_status') == 'Pending': return f"STI Follow-up"
        if row.get('condition') == 'HIV-Positive' and row.get('referral_status') == 'Pending': return "HIV Linkage Care"
        if row.get('condition') == 'Pneumonia' and pd.notna(row.get('ai_risk_score')) and row.get('ai_risk_score', 0) >= app_config.RISK_THRESHOLDS['moderate']: return f"Pneumonia (Risk: {row.get('ai_risk_score', 0):.0f})"
        if pd.notna(row.get('ai_risk_score')) and row.get('ai_risk_score', 0) >= risk_thresh: return f"High Risk ({row.get('ai_risk_score', 0):.0f})"
        return "Review Case"
        
    alerts_df['alert_reason'] = alerts_df.apply(determine_reason_chw_focused_extended, axis=1)
    alerts_df = alerts_df.sort_values(by=['ai_risk_score', 'date'], ascending=[False, False])
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)

    output_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'date', 
                   'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today']
    existing_output_cols = [col for col in output_cols if col in alerts_df.columns]
    return alerts_df[existing_output_cols]


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_summary(df_clinic_view):
    if df_clinic_view.empty: return {"tb_sputum_positivity": 0.0, "malaria_rdt_positivity": 0.0, "sti_tests_pending":0, "hiv_tests_done_period":0, "critical_disease_supply_items": 0}
    df = df_clinic_view.copy()
    for col_str in ['test_type', 'test_result', 'supply_item', 'condition']: df[col_str] = df.get(col_str, 'Unknown').astype(str)
    df['test_turnaround_days'] = pd.to_numeric(df.get('test_turnaround_days'), errors='coerce') # Used by original, can keep or remove if not displayed
    df['supply_level_days'] = pd.to_numeric(df.get('supply_level_days'), errors='coerce')

    tb_tests = df[df['test_type'].str.contains("Sputum", case=False, na=False)]
    tb_positive = tb_tests[tb_tests['test_result'] == 'Positive'].shape[0]
    tb_total_conclusive = tb_tests[~tb_tests['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])].shape[0]
    tb_sputum_positivity = (tb_positive / tb_total_conclusive) * 100 if tb_total_conclusive > 0 else 0.0

    malaria_tests = df[df['test_type'].str.contains("RDT-Malaria", case=False, na=False)]
    malaria_positive = malaria_tests[malaria_tests['test_result'] == 'Positive'].shape[0]
    malaria_total_conclusive = malaria_tests[~malaria_tests['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])].shape[0]
    malaria_rdt_positivity = (malaria_positive / malaria_total_conclusive) * 100 if malaria_total_conclusive > 0 else 0.0

    sti_tests_pending = df[(df['test_type'].isin(app_config.CRITICAL_TESTS_PENDING)) & (df['condition'].str.startswith("STI-", na=False)) & (df['test_result'] == 'Pending')].shape[0]
    hiv_tests_done_period = df[df['test_type'].str.contains("HIV", case=False, na=False) & (~df['test_result'].isin(['Pending','N/A','Unknown','nan'])) ].shape[0] # Conclusive HIV tests done

    key_disease_supplies_substrings = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Benznidazole', 'ORS', 'Oxygen']
    critical_disease_supply_items = df[(df['supply_level_days'].notna()) & (df['supply_level_days'] <= app_config.CRITICAL_SUPPLY_DAYS) & (df['supply_item'].str.contains('|'.join(key_disease_supplies_substrings), case=False, na=False))]['supply_item'].nunique()

    return {"tb_sputum_positivity": tb_sputum_positivity, "malaria_rdt_positivity": malaria_rdt_positivity, "sti_tests_pending": sti_tests_pending, "hiv_tests_done_period": hiv_tests_done_period, "critical_disease_supply_items": critical_disease_supply_items}


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_environmental_summary(df_iot_clinic_view):
    if df_iot_clinic_view.empty: return {"avg_co2": 0, "co2_alert_rooms": 0, "avg_pm25": 0, "pm25_alert_rooms": 0, "avg_occupancy":0, "high_occupancy_alert": False, "avg_sanitizer_use_hr":0}
    df = df_iot_clinic_view.copy()
    cols_to_num = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
    for col in cols_to_num: df[col] = pd.to_numeric(df.get(col), errors='coerce')
    
    avg_co2 = df.get('avg_co2_ppm', pd.Series(dtype='float')).mean(); avg_pm25 = df.get('avg_pm25', pd.Series(dtype='float')).mean(); avg_occupancy = df.get('waiting_room_occupancy', pd.Series(dtype='float')).mean(); avg_sanitizer_use_hr = df.get('sanitizer_dispenses_per_hour', pd.Series(dtype='float')).mean()
    
    latest_iot_per_room = pd.DataFrame()
    if 'timestamp' in df and 'clinic_id' in df and 'room_name' in df:
        latest_iot_per_room = df.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
    
    co2_alert_rooms = latest_iot_per_room[latest_iot_per_room.get('avg_co2_ppm', pd.Series(dtype='float')) > app_config.CO2_LEVEL_ALERT_PPM].shape[0] if not latest_iot_per_room.empty else 0
    pm25_alert_rooms = latest_iot_per_room[latest_iot_per_room.get('avg_pm25', pd.Series(dtype='float')) > app_config.PM25_ALERT_UGM3].shape[0] if not latest_iot_per_room.empty else 0
    high_occupancy_alert = (latest_iot_per_room.get('waiting_room_occupancy', pd.Series(dtype='float')) > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any() if not latest_iot_per_room.empty and latest_iot_per_room.get('waiting_room_occupancy', pd.Series(dtype='float')).notna().any() else False
    
    return {"avg_co2": avg_co2 if pd.notna(avg_co2) else 0, "co2_alert_rooms": co2_alert_rooms, "avg_pm25": avg_pm25 if pd.notna(avg_pm25) else 0, "pm25_alert_rooms": pm25_alert_rooms, "avg_occupancy": avg_occupancy if pd.notna(avg_occupancy) else 0, "high_occupancy_alert": bool(high_occupancy_alert), "avg_sanitizer_use_hr": avg_sanitizer_use_hr if pd.notna(avg_sanitizer_use_hr) else 0}


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_clinic(df_clinic_view, risk_threshold=None):
    if df_clinic_view.empty: return pd.DataFrame()
    df = df_clinic_view.copy(); df['date'] = pd.to_datetime(df.get('date'), errors='coerce'); df['test_date'] = pd.to_datetime(df.get('test_date'), errors='coerce'); df.dropna(subset=['date'], inplace=True)
    if df.empty: return pd.DataFrame()
    
    df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score'), errors='coerce')
    for col_str_ca in ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id', 'test_type']: df[col_str_ca] = df.get(col_str_ca, 'Unknown').astype(str).fillna('Unknown')
    
    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['moderate']
    latest_view_date = df['date'].max()
    recent_test_cutoff = latest_view_date - pd.Timedelta(days=7)

    cond1_ca = df['ai_risk_score'].notna() & (df['ai_risk_score'] >= risk_thresh)
    cond2_ca = (df['test_result'] == 'Positive') & df['test_date'].notna() & (df['test_date'] >= recent_test_cutoff)
    cond3_ca = (df['test_result'] == 'Pending') & (df['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS)) & (df['test_type'].isin(app_config.CRITICAL_TESTS_PENDING))
    
    alert_conditions = cond1_ca | cond2_ca | cond3_ca
    alerts_df = df[alert_conditions].copy()
    if alerts_df.empty: return pd.DataFrame()

    def determine_reason_clinic(row):
        if pd.notna(row.get('ai_risk_score')) and row['ai_risk_score'] >= risk_thresh: return f"High Risk ({row['ai_risk_score']:.0f})"
        if row.get('test_result') == 'Positive' and pd.notna(row.get('test_date')) and row['test_date'] >= recent_test_cutoff: return f"Recent Positive ({row.get('condition')})"
        if row.get('test_result') == 'Pending' and row.get('condition') in app_config.KEY_CONDITIONS_FOR_TRENDS and row.get('test_type') in app_config.CRITICAL_TESTS_PENDING: return f"Pending Critical Test ({row.get('condition')})"
        return "Review Case"
        
    alerts_df['alert_reason'] = alerts_df.apply(determine_reason_clinic, axis=1)
    alerts_df = alerts_df.sort_values(by=['ai_risk_score', 'date'], ascending=[False, False])
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    
    output_cols_ca = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result', 'referral_status', 'alert_reason', 'date']
    existing_output_cols_ca = [col for col in output_cols_ca if col in alerts_df.columns]
    return alerts_df[existing_output_cols_ca]

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_trend_data(df, value_col, date_col='date', period='D', agg_func='mean'):
    if df.empty or value_col not in df.columns or date_col not in df.columns: 
        logger.warning(f"Trend: '{value_col}' - DataFrame empty or columns missing."); 
        return pd.Series(dtype='float64').rename_axis(date_col if date_col in df.columns and df[date_col] is not None else 'date') # Ensure index has a name
    df_copy = df.copy()
    try:
        df_copy[date_col] = pd.to_datetime(df_copy.get(date_col), errors='coerce')
        if agg_func in ['mean', 'sum']: df_copy[value_col] = pd.to_numeric(df_copy.get(value_col), errors='coerce')
        
        df_copy.dropna(subset=[date_col, value_col], inplace=True)
        if df_copy.empty: 
            logger.warning(f"Trend: '{value_col}' - No valid data after NaT/NaN drop."); 
            return pd.Series(dtype='float64').rename_axis(df_copy[date_col].name if date_col in df_copy and df_copy[date_col] is not None else 'date')

        if agg_func == 'count': trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].count()
        elif agg_func == 'sum': trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].sum()
        elif agg_func == 'nunique': trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].nunique()
        else: trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].mean()
        return trend.fillna(0)
    except Exception as e: 
        logger.error(f"Trend Error for '{value_col}': {e}", exc_info=True); 
        return pd.Series(dtype='float64').rename_axis(df[date_col].name if date_col in df and df[date_col] is not None else 'date')


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_supply_forecast_data(df_health_records, forecast_days_out=7):
    # ... (This function was generally okay, ensure .get() for safety if columns might be missing) ...
    if df_health_records.empty: return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])
    df = df_health_records.copy(); df['date'] = pd.to_datetime(df.get('date'), errors='coerce'); df['supply_item'] = df.get('supply_item','Unknown').astype(str).fillna('Unknown'); df['supply_level_days'] = pd.to_numeric(df.get('supply_level_days'), errors='coerce')
    latest_supplies = df[~df['supply_item'].isin(['Unknown', 'N/A', 'nan'])].sort_values('date').drop_duplicates(subset=['supply_item'], keep='last'); latest_supplies = latest_supplies[latest_supplies['supply_level_days'].notna() & (latest_supplies['supply_level_days'] > 0)]
    if latest_supplies.empty: return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])
    forecast_list = []
    for _, row in latest_supplies.iterrows():
        item = row.get('supply_item'); current_level_days = row.get('supply_level_days'); last_recorded_date = row.get('date')
        if pd.isna(last_recorded_date) or pd.isna(current_level_days) or item is None: continue 
        daily_consumption = 1.0 
        for i in range(1, forecast_days_out + 1):
            forecast_date = last_recorded_date + pd.Timedelta(days=i); forecasted_level = max(0, current_level_days - (daily_consumption * i))
            ci_factor = 0.15; lower_ci = max(0, forecasted_level * (1 - ci_factor)); upper_ci = forecasted_level * (1 + ci_factor)
            forecast_list.append({'date': forecast_date, 'item': item, 'forecast_days': forecasted_level, 'lower_ci': lower_ci, 'upper_ci': upper_ci})
    return pd.DataFrame(forecast_list) if forecast_list else pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])


@st.cache_data(hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def get_district_summary_kpis(enriched_zone_gdf):
    # ... (This function was generally okay, ensure .get() for safety) ...
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return {"avg_population_risk": 0.0, "overall_facility_coverage": 0.0, "zones_high_risk": 0, "district_tb_burden":0, "district_malaria_burden":0, "avg_zone_co2":0.0 }
    gdf = enriched_zone_gdf.copy()
    cols_to_check = ['population', 'avg_risk_score', 'facility_coverage_score', 'active_tb_cases', 'active_malaria_cases', 'zone_avg_co2']
    for col_kpi in cols_to_check: 
        if col_kpi in gdf.columns: gdf[col_kpi] = pd.to_numeric(gdf[col_kpi], errors='coerce').fillna(0)
        else: gdf[col_kpi] = 0.0
    total_population = gdf.get('population', pd.Series(dtype='float')).sum()
    avg_pop_risk = (gdf['avg_risk_score'] * gdf['population']).sum() / total_population if total_population > 0 else 0.0
    overall_facility_coverage = (gdf['facility_coverage_score'] * gdf['population']).sum() / total_population if total_population > 0 else 0.0
    zones_high_risk = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]
    district_tb_burden = gdf.get('active_tb_cases', pd.Series(dtype='float')).sum(); district_malaria_burden = gdf.get('active_malaria_cases', pd.Series(dtype='float')).sum()
    avg_zone_co2 = gdf.get('zone_avg_co2', pd.Series(dtype='float')).mean() if gdf.get('zone_avg_co2', pd.Series(dtype='float')).notna().any() else 0.0
    return {"avg_population_risk": avg_pop_risk, "overall_facility_coverage": overall_facility_coverage, "zones_high_risk": zones_high_risk, "district_tb_burden": district_tb_burden, "district_malaria_burden": district_malaria_burden, "avg_zone_co2": avg_zone_co2}
