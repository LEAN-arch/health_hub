# utils/core_data_processing.py
import pandas as pd
import geopandas as gpd
import os
import logging
import streamlit as st
import numpy as np
from config import app_config

# Configure logging using settings from app_config
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Custom Hash Function for GeoDataFrames ---
def hash_geodataframe(gdf): # pragma: no cover
    if gdf is None: return None
    if not isinstance(gdf, gpd.GeoDataFrame):
        if isinstance(gdf, pd.DataFrame): # Handle if a non-geo DF is passed accidentally
            try: return gdf.to_parquet() # More stable hashing for DataFrames
            except Exception: return str(gdf) # Fallback for DataFrame
        return gdf # Not a DataFrame or GeoDataFrame, let Streamlit handle or raise error
    
    try:
        geom_col_name = gdf.geometry.name
        geometry_hash_part = b""
        if geom_col_name in gdf.columns and not gdf[geom_col_name].empty:
            # Ensure geometries are valid before converting to WKT for hashing
            valid_geoms = gdf[geom_col_name][gdf[geom_col_name].is_valid & ~gdf[geom_col_name].is_empty]
            if not valid_geoms.empty:
                geometry_hash_part = valid_geoms.to_wkt().values.tobytes()
        
        data_df = gdf.drop(columns=[geom_col_name], errors='ignore')
        data_hash_part = b""
        if not data_df.empty:
            data_df = data_df.reindex(sorted(data_df.columns), axis=1) # Sort cols for hash consistency
            data_hash_part = data_df.to_parquet() # Use parquet for non-geo part
            
        return (data_hash_part, geometry_hash_part)
    except Exception as e:
        logger.error(f"Error hashing GeoDataFrame (falling back to string representation): {e}", exc_info=True)
        return str(gdf)


# --- Data Loading Functions ---
@st.cache_data(ttl=3600)
def load_health_records():
    file_path = app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Health records file not found: {file_path}")
            st.error(f"Data file '{os.path.basename(file_path)}' not found. Check 'data_sources/'.")
            return pd.DataFrame()

        df = pd.read_csv(file_path, parse_dates=['date', 'referral_date', 'test_date'])

        required_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns in health records: {missing}")
            st.error(f"Health records missing required columns: {missing}. Check CSV.")
            return pd.DataFrame()

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
                df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors='coerce')

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
                df[col] = df[col].astype(str)

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
        
        if 'name' in merged_gdf.columns and merged_gdf['name'].isnull().any(): # pragma: no cover
            unmatched_geoms_in_merged = merged_gdf[merged_gdf['name'].isnull()]['zone_id'].tolist()
            if unmatched_geoms_in_merged:
                 logger.warning(f"Some zone geometries could not be matched with attributes (IDs in GeoJSON but not CSV or all attributes NaN): {unmatched_geoms_in_merged}")
        
        unmatched_attrs = zone_attributes_df[~zone_attributes_df['zone_id'].isin(merged_gdf['zone_id'])]['zone_id'].tolist()
        if unmatched_attrs: # pragma: no cover
            logger.warning(f"Some zone attributes from CSV could not be matched with geometries (IDs in CSV but not GeoJSON): {unmatched_attrs}")

        logger.info(f"Loaded and merged zone data ({len(merged_gdf)} features).")
        return merged_gdf
    except Exception as e: # pragma: no cover
        logger.error(f"Error loading zone data: {e}", exc_info=True)
        st.error(f"Error loading zone data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_iot_clinic_environment_data():
    file_path = app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"Attempting to load IoT clinic environment data from: {file_path}")
    try:
        if not os.path.exists(file_path):
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
        else: # pragma: no cover
            df['zone_id'] = 'UnknownZone'
            logger.warning("'clinic_id' not found in IoT data, 'zone_id' defaulted to 'UnknownZone'.")
        
        numeric_iot_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index',
                            'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
                            'waiting_room_occupancy', 'patient_throughput_per_hour',
                            'sanitizer_dispenses_per_hour']
        for col in numeric_iot_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: # pragma: no cover
                 logger.debug(f"Numeric IoT column '{col}' not found. Creating with NaNs.")
                 df[col] = np.nan
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
    base_cols_to_ensure_numeric = ['avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases',
                                   'pneumonia_cases', 'anemia_cases', 'sti_cases', 'ntd_chagas_cases', 'hpv_screenings_done',
                                   'avg_daily_steps_zone', 'avg_sleep_score_zone', 'avg_spo2_zone', 'avg_skin_temp_zone', 'total_falls_detected_zone',
                                   'prevalence_per_1000', 'facility_coverage_score', 'chw_visits_in_zone', # Removed total_tests_conducted as it was ambiguous
                                   'zone_avg_co2', 'zone_max_co2', 'zone_avg_temp', 'total_active_key_infections']
    base_zone_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min', 'geometry']


    if zone_gdf is None or zone_gdf.empty:
        logger.warning("Zone GeoDataFrame is empty or None for enrichment. Returning empty GDF with expected schema.")
        # Ensure expected schema for empty GDF for type consistency downstream
        current_cols = zone_gdf.columns if zone_gdf is not None and not zone_gdf.empty else []
        schema_cols = list(set(list(current_cols) + base_cols_to_ensure_numeric + base_zone_cols)) # Unique cols
        return gpd.GeoDataFrame({col: [] for col in schema_cols},
                                crs=(zone_gdf.crs if zone_gdf is not None and hasattr(zone_gdf, 'crs') else "EPSG:4326"))

    enriched_gdf = zone_gdf.copy()
    if 'zone_id' in enriched_gdf.columns:
        enriched_gdf['zone_id'] = enriched_gdf['zone_id'].astype(str)
    else: # pragma: no cover
        logger.error("CRITICAL: 'zone_id' column missing in input zone_gdf for enrichment. Returning input GDF.")
        return enriched_gdf # Cannot proceed with enrichment


    # Initialize all expected aggregate columns to 0.0 float if not present in enriched_gdf from merge
    for col in base_cols_to_ensure_numeric:
        if col not in enriched_gdf.columns: enriched_gdf[col] = 0.0

    # --- Health Data Aggregation ---
    if health_df is not None and not health_df.empty:
        health_df_copy = health_df.copy()
        if 'zone_id' not in health_df_copy.columns: # pragma: no cover
            logger.warning("Health_df missing 'zone_id', cannot perform zonal health aggregations.")
        else:
            health_df_copy['zone_id'] = health_df_copy['zone_id'].astype(str)
            
            # Define lambdas that operate on the sub-DataFrame for each group.
            def agg_count_unique_patients_condition(group_df, condition_list_str_or_list):
                if 'condition' not in group_df or 'patient_id' not in group_df : return 0
                actual_condition_list = [condition_list_str_or_list] if not isinstance(condition_list_str_or_list, list) else condition_list_str_or_list
                return group_df[group_df['condition'].astype(str).isin(actual_condition_list)]['patient_id'].nunique()

            def agg_count_sti_cases(group_df):
                if 'condition' not in group_df or 'patient_id' not in group_df: return 0
                return group_df[group_df['condition'].astype(str).str.startswith('STI-', na=False)]['patient_id'].nunique()

            def agg_hpv_screenings_done(group_df):
                if 'test_type' not in group_df: return 0
                return (group_df['test_type'].astype(str) == 'PapSmear').sum()

            zone_summary_list = []
            if 'zone_id' in health_df_copy.columns: # Ensure groupby column exists
                grouped_by_zone = health_df_copy.groupby('zone_id')
                for zone_id_val, group in grouped_by_zone: # Use zone_id_val to avoid ambiguity
                    summary_dict = {'zone_id': zone_id_val}
                    
                    # Standard aggregations from columns if they exist in the group
                    for agg_col_name, op_str in [('ai_risk_score', 'mean'), ('chw_visit', 'sum'), 
                                               ('avg_daily_steps', 'mean'), ('sleep_score_pct', 'mean'), 
                                               ('avg_spo2_pct', 'mean'), ('avg_skin_temp_celsius', 'mean'),
                                               ('fall_detected_today', 'sum')]:
                        if agg_col_name in group.columns:
                            # Ensure data is numeric for mean/sum before agg, handle all-NaN cases
                            data_to_agg = pd.to_numeric(group[agg_col_name], errors='coerce')
                            if data_to_agg.notna().any(): # Only aggregate if there's some non-NaN data
                                 summary_dict[agg_col_name] = data_to_agg.agg(op_str)
                            else: summary_dict[agg_col_name] = 0.0 if op_str == 'sum' else np.nan # mean of all NaNs is NaN
                        else: summary_dict[agg_col_name] = np.nan # Or 0 for sums

                    # Rename for zonal context
                    summary_dict['avg_daily_steps_zone'] = summary_dict.pop('avg_daily_steps', np.nan)
                    summary_dict['avg_sleep_score_zone'] = summary_dict.pop('sleep_score_pct', np.nan)
                    summary_dict['avg_spo2_zone'] = summary_dict.pop('avg_spo2_pct', np.nan)
                    summary_dict['avg_skin_temp_zone'] = summary_dict.pop('avg_skin_temp_celsius', np.nan)
                    summary_dict['total_falls_detected_zone'] = summary_dict.pop('fall_detected_today', 0) # Default to 0 if col missing


                    # Custom aggregations
                    summary_dict['active_tb_cases'] = agg_count_unique_patients_condition(group, ['TB'])
                    summary_dict['active_malaria_cases'] = agg_count_unique_patients_condition(group, ['Malaria'])
                    summary_dict['hiv_positive_cases'] = agg_count_unique_patients_condition(group, ['HIV-Positive'])
                    summary_dict['pneumonia_cases'] = agg_count_unique_patients_condition(group, ['Pneumonia'])
                    summary_dict['anemia_cases'] = agg_count_unique_patients_condition(group, ['Anemia'])
                    summary_dict['sti_cases'] = agg_count_sti_cases(group)
                    summary_dict['ntd_chagas_cases'] = agg_count_unique_patients_condition(group, ['NTD-Chagas'])
                    summary_dict['hpv_screenings_done'] = agg_hpv_screenings(group)
                    zone_summary_list.append(summary_dict)
                
                if zone_summary_list:
                    zone_summary_df = pd.DataFrame(zone_summary_list)
                    enriched_gdf = enriched_gdf.merge(zone_summary_df, on='zone_id', how='left', suffixes=('', '_from_health_agg'))
                else: # pragma: no cover
                     logger.warning("Zone summary list empty after health_df aggregation attempt (e.g. health_df was empty or had no zone_ids).")
            else: # pragma: no cover
                logger.warning("No 'zone_id' in health_df_copy for aggregation, or health_df empty.")
    
    # Fill NaNs for all health aggregates again after merge and ensure numeric types
    health_agg_cols_to_process = ['avg_risk_score', 'chw_visits_in_zone', 'active_tb_cases', 'active_malaria_cases', 
                                  'hiv_positive_cases', 'pneumonia_cases', 'anemia_cases', 'sti_cases', 
                                  'ntd_chagas_cases', 'hpv_screenings_done', 'avg_daily_steps_zone', 
                                  'avg_sleep_score_zone', 'avg_spo2_zone', 'avg_skin_temp_zone', 'total_falls_detected_zone']
    for col in health_agg_cols_to_process:
        suffixed_col = f"{col}_from_health_agg"
        target_col_val = enriched_gdf[suffixed_col] if suffixed_col in enriched_gdf.columns else enriched_gdf.get(col) # Use original if no suffix from merge
        
        enriched_gdf[col] = pd.to_numeric(target_col_val, errors='coerce').fillna(0)
        if suffixed_col in enriched_gdf.columns and suffixed_col != col : 
            enriched_gdf.drop(columns=[suffixed_col], inplace=True, errors='ignore')


    # --- IoT Data Aggregation ---
    if iot_df is not None and not iot_df.empty:
        iot_df_copy = iot_df.copy()
        if 'zone_id' not in iot_df_copy.columns or iot_df_copy['zone_id'].isnull().all() or (iot_df_copy['zone_id']=='UnknownZone').all(): # pragma: no cover
            logger.warning("zone_id missing or all UnknownZone in IoT data for enrichment by zone.")
        else:
            iot_df_copy['zone_id'] = iot_df_copy['zone_id'].astype(str)
            iot_numeric_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_temp_celsius'] # For aggregation
            for iot_col_num in iot_numeric_cols:
                if iot_col_num in iot_df_copy.columns: iot_df_copy[iot_col_num] = pd.to_numeric(iot_df_copy[iot_col_num], errors='coerce')
                else: iot_df_copy[iot_col_num] = np.nan # pragma: no cover
            
            latest_iot_zone = iot_df_copy.sort_values('timestamp').drop_duplicates(subset=['zone_id', 'room_name'], keep='last')
            if not latest_iot_zone.empty:
                # Aggregate only if columns exist and have some non-NA data for the group
                agg_dict_iot = {}
                if 'avg_co2_ppm' in latest_iot_zone.columns and latest_iot_zone['avg_co2_ppm'].notna().any():
                    agg_dict_iot['zone_avg_co2'] = ('avg_co2_ppm', 'mean')
                if 'max_co2_ppm' in latest_iot_zone.columns and latest_iot_zone['max_co2_ppm'].notna().any():
                    agg_dict_iot['zone_max_co2'] = ('max_co2_ppm', 'max')
                if 'avg_temp_celsius' in latest_iot_zone.columns and latest_iot_zone['avg_temp_celsius'].notna().any():
                    agg_dict_iot['zone_avg_temp'] = ('avg_temp_celsius', 'mean')

                if agg_dict_iot: # If there's anything to aggregate
                    iot_zone_summary = latest_iot_zone.groupby('zone_id').agg(**agg_dict_iot).reset_index()
                    enriched_gdf = enriched_gdf.merge(iot_zone_summary, on='zone_id', how='left', suffixes=('', '_from_iot_agg'))
                else: # pragma: no cover
                    logger.info("No IoT data columns found suitable for aggregation by zone.")
            else: # pragma: no cover
                 logger.info("No IoT data to aggregate by zone after filtering for latest readings.")

    expected_iot_agg_cols = ['zone_avg_co2', 'zone_max_co2', 'zone_avg_temp']
    for col in expected_iot_agg_cols:
        suffixed_col = f"{col}_from_iot_agg"
        target_col_val = enriched_gdf[suffixed_col] if suffixed_col in enriched_gdf.columns else enriched_gdf.get(col)
        
        enriched_gdf[col] = pd.to_numeric(target_col_val, errors='coerce').fillna(0)
        if suffixed_col in enriched_gdf.columns and suffixed_col != col:
            enriched_gdf.drop(columns=[suffixed_col], inplace=True, errors='ignore')


    # --- Recalculate total_active_key_infections and prevalence_per_1000 ---
    key_infectious_cols = ['active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'sti_cases']
    enriched_gdf['total_active_key_infections'] = 0.0
    for col_inf in key_infectious_cols:
        if col_inf in enriched_gdf.columns: enriched_gdf['total_active_key_infections'] += pd.to_numeric(enriched_gdf[col_inf], errors='coerce').fillna(0)
        else: enriched_gdf[col_inf] = 0.0 # Ensure column exists for safety

    if 'population' in enriched_gdf.columns and 'total_active_key_infections' in enriched_gdf.columns:
        enriched_gdf['population'] = pd.to_numeric(enriched_gdf['population'], errors='coerce').fillna(0)
        enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(lambda row: (row['total_active_key_infections'] / row['population']) * 1000 if row['population'] > 0 else 0.0, axis=1)
    else: enriched_gdf['prevalence_per_1000'] = 0.0
    
    # --- Recalculate facility_coverage_score ---
    if 'avg_travel_time_clinic_min' in enriched_gdf.columns and 'num_clinics' in enriched_gdf.columns:
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

    # Final fillna for key score columns created in this function
    for score_col_final in ['travel_score', 'clinic_count_score', 'facility_coverage_score', 'prevalence_per_1000']:
        if score_col_final not in enriched_gdf.columns: enriched_gdf[score_col_final] = 0.0 # Ensure col exists
        enriched_gdf[score_col_final] = pd.to_numeric(enriched_gdf[score_col_final], errors='coerce').fillna(
            enriched_gdf[score_col_final].median() if pd.api.types.is_numeric_dtype(enriched_gdf[score_col_final]) and not enriched_gdf[score_col_final].empty and enriched_gdf[score_col_final].notna().any() else 0.0)

    logger.info("Zone geodata successfully enriched.")
    return enriched_gdf


# --- KPI Calculation Functions ---
# (All these functions should be the robust versions from the "Complete and Corrected for Sensor/Disease Focus" response)
@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_overall_kpis(df_health_records):
    if df_health_records.empty: return {"total_patients": 0, "tb_cases_active": 0, "malaria_rdt_positive_rate":0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals": 0}
    df = df_health_records.copy(); 
    if 'date' not in df.columns or df['date'].isnull().all(): return {"total_patients": 0, "tb_cases_active": 0, "malaria_rdt_positive_rate":0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals": 0}
    df['date'] = pd.to_datetime(df['date'], errors='coerce'); df.dropna(subset=['date'], inplace=True)
    if df.empty: return {"total_patients": 0, "tb_cases_active": 0, "malaria_rdt_positive_rate":0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals": 0}
    latest_date = df['date'].max(); period_start_date = latest_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND)
    for col in ['patient_id', 'condition', 'test_result', 'referral_status', 'test_type']: df[col] = df.get(col, pd.Series(dtype='str')).astype(str).fillna('Unknown')
    df['test_date'] = pd.to_datetime(df.get('test_date'), errors='coerce')
    total_patients = df['patient_id'].nunique(); tb_cases_active = df[(df['condition'] == 'TB') & (df['test_result'] == 'Positive') & (df['referral_status'] != 'Completed')]['patient_id'].nunique()
    malaria_rdt_tests = df[df['test_type'] == 'RDT-Malaria']; malaria_rdt_positive = malaria_rdt_tests[malaria_rdt_tests['test_result'] == 'Positive'].shape[0]; malaria_rdt_total_conclusive = malaria_rdt_tests[~malaria_rdt_tests['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])].shape[0]; malaria_rdt_positive_rate = (malaria_rdt_positive / malaria_rdt_total_conclusive) * 100 if malaria_rdt_total_conclusive > 0 else 0.0
    hiv_newly_diagnosed = df[(df['condition'] == 'HIV-Positive') & (df['test_result'] == 'Positive') & (df['test_type'].str.contains("HIV", case=False, na=False)) & (df['test_date'].notna()) & (df['test_date'] >= period_start_date)]['patient_id'].nunique()
    pending_critical_referrals = df[(df['referral_status'] == 'Pending') & (df['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS))]['patient_id'].nunique()
    return {"total_patients": total_patients, "tb_cases_active": tb_cases_active, "malaria_rdt_positive_rate": malaria_rdt_positive_rate, "hiv_newly_diagnosed_period": hiv_newly_diagnosed, "pending_critical_referrals": pending_critical_referrals}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_chw_summary(df_chw_view):
    if df_chw_view.empty: return {"visits_today": 0, "tb_contacts_to_trace": 0, "sti_symptomatic_referrals": 0, "avg_patient_risk_visited":0.0, "patients_low_spo2_visited": 0, "patients_fever_visited": 0, "avg_chw_steps":0, "avg_chw_sleep_score":0} # Added new KPIs
    df = df_chw_view.copy()
    df['chw_visit'] = pd.to_numeric(df.get('chw_visit'), errors='coerce').fillna(0); df['condition'] = df.get('condition', 'Unknown').astype(str); df['tb_contact_traced'] = pd.to_numeric(df.get('tb_contact_traced'), errors='coerce').fillna(0); df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score'), errors='coerce'); df['referral_status'] = df.get('referral_status', 'Unknown').astype(str); df['avg_spo2_pct'] = pd.to_numeric(df.get('avg_spo2_pct'), errors='coerce'); df['max_skin_temp_celsius'] = pd.to_numeric(df.get('max_skin_temp_celsius'), errors='coerce')
    df['avg_daily_steps'] = pd.to_numeric(df.get('avg_daily_steps'), errors='coerce') # New
    df['sleep_score_pct'] = pd.to_numeric(df.get('sleep_score_pct'), errors='coerce') # New

    visits_today = df[df['chw_visit'] == 1].shape[0]
    tb_contacts_to_trace = df[(df['condition'] == 'TB') & (df['tb_contact_traced'] == 0)].shape[0]
    sti_symptomatic_referrals = df[(df['condition'].str.startswith("STI-", na=False)) & (df['referral_status'] == 'Pending')].shape[0]
    patients_visited_today_df = df[df['chw_visit'] == 1]
    avg_patient_risk_visited = patients_visited_today_df['ai_risk_score'].mean() if not patients_visited_today_df.empty and patients_visited_today_df['ai_risk_score'].notna().any() else 0.0
    patients_low_spo2_visited = patients_visited_today_df[patients_visited_today_df['avg_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT].shape[0] if 'avg_spo2_pct' in patients_visited_today_df and patients_visited_today_df['avg_spo2_pct'].notna().any() else 0
    patients_fever_visited = patients_visited_today_df[patients_visited_today_df['max_skin_temp_celsius'] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C].shape[0] if 'max_skin_temp_celsius' in patients_visited_today_df and patients_visited_today_df['max_skin_temp_celsius'].notna().any() else 0
    # Assuming these are averages of patients visited today for CHW context, or could be CHW's own data if a CHW ID was used to filter
    avg_steps_calc = patients_visited_today_df['avg_daily_steps'].mean() if not patients_visited_today_df.empty and 'avg_daily_steps' in patients_visited_today_df and patients_visited_today_df['avg_daily_steps'].notna().any() else 0
    avg_sleep_score_calc = patients_visited_today_df['sleep_score_pct'].mean() if not patients_visited_today_df.empty and 'sleep_score_pct' in patients_visited_today_df and patients_visited_today_df['sleep_score_pct'].notna().any() else 0

    return {"visits_today": visits_today, "tb_contacts_to_trace": tb_contacts_to_trace, "sti_symptomatic_referrals": sti_symptomatic_referrals, "avg_patient_risk_visited": avg_patient_risk_visited if pd.notna(avg_patient_risk_visited) else 0.0, "patients_low_spo2_visited": patients_low_spo2_visited, "patients_fever_visited": patients_fever_visited, "avg_chw_steps": avg_steps_calc if pd.notna(avg_steps_calc) else 0, "avg_chw_sleep_score": avg_sleep_score_calc if pd.notna(avg_sleep_score_calc) else 0}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_chw(df_chw_view, risk_threshold=None):
    if df_chw_view.empty: return pd.DataFrame()
    df_copy = df_chw_view.copy(); df_copy['date'] = pd.to_datetime(df_copy.get('date'), errors='coerce'); df_copy['test_date'] = pd.to_datetime(df_copy.get('test_date'), errors='coerce'); df_copy.dropna(subset=['date'], inplace=True)
    if df_copy.empty: return pd.DataFrame()
    numeric_cols_chw_alert_nan = ['ai_risk_score', 'min_spo2_pct', 'max_skin_temp_celsius']; numeric_cols_chw_alert_zero = ['tb_contact_traced', 'fall_detected_today']; string_cols_chw_alert_unknown = ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id']
    for col in numeric_cols_chw_alert_nan: 
        if col not in df_copy.columns: df_copy[col] = np.nan
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    for col in numeric_cols_chw_alert_zero: 
        if col not in df_copy.columns: df_copy[col] = 0
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    for col in string_cols_chw_alert_unknown: 
        if col not in df_copy.columns: df_copy[col] = 'Unknown'
        df_copy[col] = df_copy[col].astype(str).fillna('Unknown')
    risk_thresh = risk_threshold if risk_threshold is not None else app_config.RISK_THRESHOLDS['chw_alert_moderate']
    cond_risk = df_copy['ai_risk_score'].notna() & (df_copy['ai_risk_score'] >= risk_thresh)
    cond_tb_trace = (df_copy['condition'] == 'TB') & (df_copy['tb_contact_traced'] == 0)
    cond_tb_referral = (df_copy['condition'] == 'TB') & (df_copy['referral_status'] == 'Pending')
    cond_sti_referral = df_copy['condition'].str.startswith("STI-", na=False) & (df_copy['referral_status'] == 'Pending')
    cond_hiv_referral = (df_copy['condition'] == 'HIV-Positive') & (df_copy['referral_status'] == 'Pending')
    cond_pneumonia_risk = (df_copy['condition'] == 'Pneumonia') & df_copy['ai_risk_score'].notna() & (df_copy['ai_risk_score'] >= app_config.RISK_THRESHOLDS['moderate'])
    cond_low_spo2 = df_copy['min_spo2_pct'].notna() & (df_copy['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT)
    cond_fever = df_copy['max_skin_temp_celsius'].notna() & (df_copy['max_skin_temp_celsius'] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C)
    cond_fall = df_copy['fall_detected_today'] > 0
    alert_conditions = cond_risk | cond_tb_trace | cond_tb_referral | cond_sti_referral | cond_hiv_referral | cond_pneumonia_risk | cond_low_spo2 | cond_fever | cond_fall
    alerts_df = df_copy[alert_conditions].copy(); 
    if alerts_df.empty: return pd.DataFrame()
    def determine_reason_chw_focused_extended(row):
        # Prioritize more critical/specific alerts first
        if pd.notna(row.get('fall_detected_today')) and row.get('fall_detected_today', 0) > 0: return f"Fall Detected"
        if pd.notna(row.get('min_spo2_pct')) and row.get('min_spo2_pct', 101) < app_config.SPO2_LOW_THRESHOLD_PCT: return f"Low SpO2 ({row.get('min_spo2_pct', 0):.0f}%)"
        if pd.notna(row.get('max_skin_temp_celsius')) and row.get('max_skin_temp_celsius', 0) >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C: return f"Fever ({row.get('max_skin_temp_celsius', 0):.1f}°C)"
        if row.get('condition') == 'TB' and row.get('tb_contact_traced', 1) == 0: return "TB Contact Trace"
        if row.get('condition') == 'TB' and row.get('referral_status') == 'Pending': return "TB Referral Pend."
        if str(row.get('condition','')).startswith("STI-") and row.get('referral_status') == 'Pending': return f"STI Follow-up ({row.get('condition')})"
        if row.get('condition') == 'HIV-Positive' and row.get('referral_status') == 'Pending': return "HIV Linkage Care"
        if row.get('condition') == 'Pneumonia' and pd.notna(row.get('ai_risk_score')) and row.get('ai_risk_score', 0) >= app_config.RISK_THRESHOLDS['moderate']: return f"Pneumonia (Risk: {row.get('ai_risk_score', 0):.0f})"
        if pd.notna(row.get('ai_risk_score')) and row.get('ai_risk_score', 0) >= risk_thresh: return f"High Risk ({row.get('ai_risk_score', 0):.0f})"
        return "Review Case" # Fallback
    alerts_df['alert_reason'] = alerts_df.apply(determine_reason_chw_focused_extended, axis=1); alerts_df = alerts_df.sort_values(by=['ai_risk_score', 'date'], ascending=[False, False]); alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    output_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'date', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today']; existing_output_cols = [col for col in output_cols if col in alerts_df.columns]
    return alerts_df[existing_output_cols]

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_summary(df_clinic_view):
    if df_clinic_view.empty: return {"tb_sputum_positivity": 0.0, "malaria_rdt_positivity": 0.0, "sti_tests_pending":0, "hiv_tests_done_period":0, "critical_disease_supply_items": 0}
    df = df_clinic_view.copy(); 
    for col_str in ['test_type', 'test_result', 'supply_item', 'condition']: df[col_str] = df.get(col_str, pd.Series(dtype='str')).astype(str).fillna('Unknown')
    # df['test_turnaround_days'] = pd.to_numeric(df.get('test_turnaround_days'), errors='coerce') # Retained if used for general TAT KPI
    df['supply_level_days'] = pd.to_numeric(df.get('supply_level_days'), errors='coerce')

    tb_tests = df[df['test_type'].str.contains("Sputum|GeneXpert", case=False, na=False)] # Broaden TB test types
    tb_positive = tb_tests[tb_tests['test_result'] == 'Positive'].shape[0]
    tb_total_conclusive = tb_tests[~tb_tests['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])].shape[0]
    tb_sputum_positivity = (tb_positive / tb_total_conclusive) * 100 if tb_total_conclusive > 0 else 0.0

    malaria_tests = df[df['test_type'].str.contains("RDT-Malaria|Microscopy-Malaria", case=False, na=False)]
    malaria_positive = malaria_tests[malaria_tests['test_result'] == 'Positive'].shape[0]
    malaria_total_conclusive = malaria_tests[~malaria_tests['test_result'].isin(['Pending', 'N/A', 'Unknown', 'nan'])].shape[0]
    malaria_rdt_positivity = (malaria_positive / malaria_total_conclusive) * 100 if malaria_total_conclusive > 0 else 0.0

    sti_tests_pending = df[(df['test_type'].isin(app_config.CRITICAL_TESTS_PENDING)) & (df['condition'].str.startswith("STI-", na=False)) & (df['test_result'] == 'Pending')].shape[0]
    hiv_tests_done_period = df[df['test_type'].str.contains("HIV", case=False, na=False) & (~df['test_result'].isin(['Pending','N/A','Unknown','nan'])) ].shape[0]
    
    key_disease_supplies_substrings = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Benznidazole', 'ORS', 'Oxygen', 'Metronidazole', 'Gardasil']
    critical_disease_supply_items = df[(df['supply_level_days'].notna()) & (df['supply_level_days'] <= app_config.CRITICAL_SUPPLY_DAYS) & (df['supply_item'].str.contains('|'.join(key_disease_supplies_substrings), case=False, na=False))]['supply_item'].nunique()

    return {"tb_sputum_positivity": tb_sputum_positivity, "malaria_rdt_positivity": malaria_rdt_positivity, "sti_tests_pending": sti_tests_pending, "hiv_tests_done_period": hiv_tests_done_period, "critical_disease_supply_items": critical_disease_supply_items}


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_environmental_summary(df_iot_clinic_view):
    if df_iot_clinic_view.empty: return {"avg_co2": 0, "co2_alert_rooms": 0, "avg_pm25": 0, "pm25_alert_rooms": 0, "avg_occupancy":0, "high_occupancy_alert": False, "avg_sanitizer_use_hr":0}
    df = df_iot_clinic_view.copy()
    cols_to_num_iot = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
    for col_iot in cols_to_num_iot: df[col_iot] = pd.to_numeric(df.get(col_iot), errors='coerce') # Use .get()
    
    avg_co2 = df.get('avg_co2_ppm', pd.Series(dtype='float')).mean(); avg_pm25 = df.get('avg_pm25', pd.Series(dtype='float')).mean(); avg_occupancy = df.get('waiting_room_occupancy', pd.Series(dtype='float')).mean(); avg_sanitizer_use_hr = df.get('sanitizer_dispenses_per_hour', pd.Series(dtype='float')).mean()
    
    latest_iot_per_room = pd.DataFrame() # Default to empty
    if 'timestamp' in df and 'clinic_id' in df and 'room_name' in df:
        latest_iot_per_room = df.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
    
    co2_alert_rooms = 0; pm25_alert_rooms = 0; high_occupancy_alert = False
    if not latest_iot_per_room.empty:
        if 'avg_co2_ppm' in latest_iot_per_room: co2_alert_rooms = latest_iot_per_room[latest_iot_per_room['avg_co2_ppm'] > app_config.CO2_LEVEL_ALERT_PPM].shape[0]
        if 'avg_pm25' in latest_iot_per_room: pm25_alert_rooms = latest_iot_per_room[latest_iot_per_room['avg_pm25'] > app_config.PM25_ALERT_UGM3].shape[0]
        if 'waiting_room_occupancy' in latest_iot_per_room and latest_iot_per_room['waiting_room_occupancy'].notna().any():
            high_occupancy_alert = (latest_iot_per_room['waiting_room_occupancy'] > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            
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
    cond2_ca = (df.get('test_result', pd.Series(dtype=str)) == 'Positive') & df.get('test_date', pd.Series(dtype='datetime64[ns]')).notna() & (df.get('test_date', pd.Series(dtype='datetime64[ns]')) >= recent_test_cutoff)
    cond3_ca = (df.get('test_result', pd.Series(dtype=str)) == 'Pending') & (df.get('condition', pd.Series(dtype=str)).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)) & (df.get('test_type', pd.Series(dtype=str)).isin(app_config.CRITICAL_TESTS_PENDING))
    
    alert_conditions = cond1_ca | cond2_ca | cond3_ca
    alerts_df = df[alert_conditions].copy()
    if alerts_df.empty: return pd.DataFrame()

    def determine_reason_clinic(row): # Ensure row.get() for safety
        if pd.notna(row.get('ai_risk_score')) and row.get('ai_risk_score', 0) >= risk_thresh: return f"High Risk ({row['ai_risk_score']:.0f})"
        if row.get('test_result') == 'Positive' and pd.notna(row.get('test_date')) and row.get('test_date', pd.Timestamp.min) >= recent_test_cutoff: return f"Recent Positive ({row.get('condition')})"
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
    # Use .get() for accessing columns and provide a default if the column doesn't exist
    # This makes the function more robust if df is missing expected columns.
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        logger.warning(f"Trend data generation skipped for '{value_col}' due to missing data or columns.")
        return pd.Series(dtype='float64').rename_axis(df.get(date_col, pd.Series(name='date',dtype='datetime64[ns]')).name)


    df_copy = df.copy()
    try:
        # Ensure date_col is datetime and exists
        df_copy[date_col] = pd.to_datetime(df_copy.get(date_col), errors='coerce')
        
        # Ensure value_col is numeric for relevant aggregations
        if agg_func in ['mean', 'sum']:
            df_copy[value_col] = pd.to_numeric(df_copy.get(value_col), errors='coerce')
        
        df_copy.dropna(subset=[date_col, value_col], inplace=True) # Crucial step

        if df_copy.empty:
            logger.warning(f"Trend: '{value_col}' - No valid data points after NaT/NaN drop.")
            return pd.Series(dtype='float64').rename_axis(df_copy[date_col].name if date_col in df_copy else 'date')

        if agg_func == 'count': trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].count()
        elif agg_func == 'sum': trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].sum()
        elif agg_func == 'nunique': trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].nunique()
        else: trend = df_copy.groupby(pd.Grouper(key=date_col, freq=period))[value_col].mean()
        
        return trend.fillna(0)
    except Exception as e: # pragma: no cover
        logger.error(f"Trend Error for '{value_col} on {date_col}': {e}", exc_info=True)
        return pd.Series(dtype='float64').rename_axis(df.get(date_col, pd.Series(name='date',dtype='datetime64[ns]')).name)


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_supply_forecast_data(df_health_records, forecast_days_out=7):
    if df_health_records.empty: return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])
    df = df_health_records.copy()
    df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
    df['supply_item'] = df.get('supply_item','Unknown').astype(str).fillna('Unknown')
    df['supply_level_days'] = pd.to_numeric(df.get('supply_level_days'), errors='coerce')

    latest_supplies = df[~df['supply_item'].isin(['Unknown', 'N/A', 'nan'])].sort_values('date').drop_duplicates(subset=['supply_item'], keep='last')
    latest_supplies = latest_supplies[latest_supplies['supply_level_days'].notna() & (latest_supplies['supply_level_days'] > 0)]
    
    if latest_supplies.empty: return pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])
    
    forecast_list = []
    for _, row in latest_supplies.iterrows():
        item = row.get('supply_item'); current_level_days = row.get('supply_level_days'); last_recorded_date = row.get('date')
        if pd.isna(last_recorded_date) or pd.isna(current_level_days) or item is None: continue 
        daily_consumption = 1.0 # Simplified assumption
        for i in range(1, forecast_days_out + 1):
            forecast_date = last_recorded_date + pd.Timedelta(days=i)
            forecasted_level = max(0, current_level_days - (daily_consumption * i))
            ci_factor = 0.15; lower_ci = max(0, forecasted_level * (1 - ci_factor)); upper_ci = forecasted_level * (1 + ci_factor)
            forecast_list.append({'date': forecast_date, 'item': item, 'forecast_days': forecasted_level, 'lower_ci': lower_ci, 'upper_ci': upper_ci})
            
    return pd.DataFrame(forecast_list) if forecast_list else pd.DataFrame(columns=['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci'])


@st.cache_data(hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def get_district_summary_kpis(enriched_zone_gdf):
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return {"avg_population_risk": 0.0, "overall_facility_coverage": 0.0, "zones_high_risk": 0, "district_tb_burden":0, "district_malaria_burden":0, "avg_zone_co2":0.0 }
    
    gdf = enriched_zone_gdf.copy()
    cols_to_check_dist = { # Colname: default_value if missing
        'population': 0.0, 'avg_risk_score': 0.0, 'facility_coverage_score': 0.0, 
        'active_tb_cases': 0.0, 'active_malaria_cases': 0.0, 'zone_avg_co2': 0.0
    }
    for col_kpi_dist, default_val in cols_to_check_dist.items(): 
        gdf[col_kpi_dist] = pd.to_numeric(gdf.get(col_kpi_dist, default_val), errors='coerce').fillna(0)

    total_population = gdf.get('population', pd.Series(dtype='float')).sum() # Summing a Series
    
    avg_pop_risk = (gdf['avg_risk_score'] * gdf['population']).sum() / total_population if total_population > 0 else 0.0
    overall_facility_coverage = (gdf['facility_coverage_score'] * gdf['population']).sum() / total_population if total_population > 0 else 0.0
    zones_high_risk = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]
    district_tb_burden = gdf.get('active_tb_cases', pd.Series(dtype='float')).sum()
    district_malaria_burden = gdf.get('active_malaria_cases', pd.Series(dtype='float')).sum()
    avg_zone_co2_series = gdf.get('zone_avg_co2', pd.Series(dtype='float'))
    avg_zone_co2 = avg_zone_co2_series.mean() if avg_zone_co2_series.notna().any() else 0.0
    
    return {"avg_population_risk": avg_pop_risk, "overall_facility_coverage": overall_facility_coverage, 
            "zones_high_risk": zones_high_risk, "district_tb_burden": district_tb_burden, 
            "district_malaria_burden": district_malaria_burden, "avg_zone_co2": avg_zone_co2}
