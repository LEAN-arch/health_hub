# health_hub/utils/core_data_processing.py
import pandas as pd
import geopandas as gpd
import os
import logging
import streamlit as st
import numpy as np
from config import app_config # app_config needs to be in PYTHONPATH or same dir

logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# --- Custom Hash Function for GeoDataFrames ---
def hash_geodataframe(gdf): # pragma: no cover
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return None # Or some other stable hash for None/empty
    try:
        # Sort by index and columns for consistent hashing
        gdf_sorted = gdf.sort_index().reindex(sorted(gdf.columns), axis=1)
        # Hash non-geometry part using parquet (more stable than to_string or to_json)
        data_hash_part = gdf_sorted.drop(columns=[gdf_sorted.geometry.name], errors='ignore').to_parquet()
        # Hash geometry part using WKB (Well-Known Binary)
        # Ensure geometries are valid before converting to WKB
        valid_geoms = gdf_sorted[gdf_sorted.geometry.name][gdf_sorted[gdf_sorted.geometry.name].is_valid & ~gdf_sorted[gdf_sorted.geometry.name].is_empty]
        geometry_hash_part = b""
        if not valid_geoms.empty:
            geometry_hash_part = valid_geoms.to_wkb().values.tobytes()
        # Include CRS in the hash as it affects geometry representation
        crs_hash_part = gdf_sorted.crs.to_wkt() if gdf_sorted.crs else None
        return (data_hash_part, geometry_hash_part, crs_hash_part)
    except Exception as e:
        logger.error(f"Error hashing GeoDataFrame (falling back to basic hash of head): {e}", exc_info=True)
        # Fallback: hash of string representation of WKT for geometries and values for other columns
        return (
            str(gdf.drop(columns=[gdf.geometry.name], errors='ignore').head(3).to_dict()),
            str(gdf.geometry.to_wkt().head(3).to_dict()),
            str(gdf.crs.to_wkt() if gdf.crs else None)
        )

# --- Data Loading Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_health_records():
    file_path = app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Health records file not found: {file_path}")
            st.error(f"Data file '{os.path.basename(file_path)}' not found. Please check 'data_sources/' directory and configuration.")
            return pd.DataFrame() # Return empty DataFrame with expected schema for graceful failure

        df = pd.read_csv(file_path, low_memory=False) # low_memory=False for mixed types initial read

        # Date parsing - critical for time-series analysis
        date_cols_to_parse = ['date', 'referral_date', 'test_date']
        for col in date_cols_to_parse:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                logger.warning(f"Date column '{col}' not found in health_records.csv. It will be missing or handled as NaT if created later.")

        # Ensure essential columns exist - APP WILL LIKELY FAIL WITHOUT THESE
        required_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score"]
        missing_req = [col for col in required_cols if col not in df.columns or df[col].isnull().all()] # Check if column exists AND has non-null data
        if missing_req:
            logger.error(f"Missing or entirely null critical required columns in health records: {missing_req}. Cannot robustly proceed.")
            st.error(f"Health records are missing critical data in columns: {missing_req}. Please check the CSV file structure and content.")
            return pd.DataFrame() # Return empty with schema might be better than just empty

        # Define expected numeric and string columns based on comprehensive use
        # This helps ensure correct dtypes for calculations and plotting
        numeric_cols_expected = [
            'test_turnaround_days', 'quantity_dispensed', 'stock_on_hand', 'consumption_rate_per_day',
            'ai_risk_score', 'avg_daily_steps', 'avg_spo2', 'min_spo2_pct', 'max_skin_temp_celsius',
            'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced', 'resting_heart_rate',
            'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score',
            'hiv_viral_load'
        ]
        string_cols_expected = [
            'patient_id', 'condition', 'test_type', 'test_result', 'item',
            'zone_id', 'clinic_id', 'physician_id', 'notes', 'hpv_status',
            'referral_status', 'gender'
        ]

        for col in numeric_cols_expected:
            if col not in df.columns:
                logger.debug(f"Numeric column '{col}' not found in health_records.csv. Creating as NaN-filled numeric column.")
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce ensures conversion, invalid parsing becomes NaN

        for col in string_cols_expected:
            if col not in df.columns:
                logger.debug(f"String column '{col}' not found in health_records.csv. Creating as 'Unknown'-filled string column.")
                df[col] = 'Unknown'
            # Robust fillna for strings and convert to string type
            df[col] = df[col].astype(str).fillna('Unknown')
            # Replace common null/empty representations with 'Unknown'
            common_null_strs = ['', 'nan', 'None', 'NONE', 'Null', 'NULL', '<NA>'] # Add any other observed "empty" strings
            df.loc[df[col].isin(common_null_strs), col] = 'Unknown'

        # Specific cleaning for key categorical columns after ensuring they are string
        df['condition'] = df['condition'].replace('Healthy Checkup', 'Wellness Visit')
        # Add more specific cleaning as needed (e.g., standardizing test_result values)
        df['test_result'] = df['test_result'].replace({'Positive ': 'Positive', ' Negative': 'Negative'}) # Trim spaces

        logger.info(f"Successfully loaded and preprocessed health records from {file_path} ({len(df)} rows).")
        return df
    except pd.errors.EmptyDataError: # Specific error for empty file
        logger.error(f"Health records file is empty: {file_path}")
        st.error(f"Health records file '{os.path.basename(file_path)}' is empty.")
        return pd.DataFrame()
    except Exception as e: # Catch-all for other loading/parsing errors
        logger.error(f"Unexpected error loading health records from {file_path}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while loading health records: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def load_zone_data():
    attributes_path = app_config.ZONE_ATTRIBUTES_CSV
    geometries_path = app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"Loading zone attributes from: {attributes_path}")
    logger.info(f"Loading zone geometries from: {geometries_path}")

    if not os.path.exists(attributes_path):
        logger.error(f"Zone attributes file not found: {attributes_path}")
        st.error(f"Zone attributes file ('{os.path.basename(attributes_path)}') not found.")
        return None # Indicate critical failure
    if not os.path.exists(geometries_path):
        logger.error(f"Zone geometries file not found: {geometries_path}")
        st.error(f"Zone geometries file ('{os.path.basename(geometries_path)}') not found.")
        return None

    try:
        zone_attributes_df = pd.read_csv(attributes_path)
        required_attr_cols = ["zone_id", "name", "population", "socio_economic_index", "num_clinics", "avg_travel_time_clinic_min"]
        missing_attrs = [col for col in required_attr_cols if col not in zone_attributes_df.columns]
        if missing_attrs:
            logger.error(f"Zone attributes CSV is missing required columns: {missing_attrs}.")
            st.error(f"Zone attributes CSV is missing required columns: {missing_attrs}. Please check the file.")
            return None

        zone_geometries_gdf = gpd.read_file(geometries_path)
        if "zone_id" not in zone_geometries_gdf.columns:
            logger.error("Zone geometries GeoJSON is missing the 'zone_id' property in features.")
            st.error("Zone geometries GeoJSON is missing the 'zone_id' property. Cannot merge with attributes.")
            return None

        # Standardize CRS
        if zone_geometries_gdf.crs is None:
             logger.warning(f"Zone geometries GeoJSON has no CRS defined. Assuming {app_config.DEFAULT_CRS}.")
             zone_geometries_gdf = zone_geometries_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif zone_geometries_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): # Case-insensitive check
             logger.info(f"Reprojecting zone geometries from {zone_geometries_gdf.crs.to_string()} to {app_config.DEFAULT_CRS}.")
             zone_geometries_gdf = zone_geometries_gdf.to_crs(app_config.DEFAULT_CRS)

        # Ensure zone_id is string type for reliable merging
        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str)
        zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str)

        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left")

        # Check for merge issues: zones in geojson not in attributes, or vice-versa
        if merged_gdf['name'].isnull().any(): # 'name' is a good proxy from attributes
            unmatched_geozones = merged_gdf[merged_gdf['name'].isnull()]['zone_id'].unique().tolist()
            logger.warning(f"Some zone_ids from GeoJSON were not found in attributes CSV or had null attributes: {unmatched_geozones}. These zones will have missing attribute data.")
            # Fill missing names with zone_id for basic identification
            merged_gdf.loc[merged_gdf['name'].isnull(), 'name'] = "Zone " + merged_gdf['zone_id']

        # Ensure numeric types for relevant attribute columns after merge, fill NaNs appropriately
        numeric_attr_cols = ["population", "socio_economic_index", "num_clinics", "avg_travel_time_clinic_min"]
        for col in numeric_attr_cols:
            if col in merged_gdf.columns:
                merged_gdf[col] = pd.to_numeric(merged_gdf[col], errors='coerce').fillna(0) # Fill with 0 if conversion fails or is NaN
            else: # Should not happen if required_attr_cols check passed
                logger.warning(f"Numeric attribute column '{col}' missing post-merge. Initializing to 0.")
                merged_gdf[col] = 0.0

        logger.info(f"Successfully loaded and merged zone data, resulting in {len(merged_gdf)} zone features.")
        return merged_gdf

    except Exception as e: # Catch-all for other loading/parsing errors
        logger.error(f"Error loading or merging zone data: {e}", exc_info=True)
        st.error(f"An error occurred while loading zone data: {e}")
        return None


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_iot_clinic_environment_data():
    file_path = app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"Attempting to load IoT clinic environment data from: {file_path}")
    default_empty_iot_df = pd.DataFrame(columns=['timestamp', 'clinic_id', 'room_name', 'zone_id', 'avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'sanitizer_dispenses_per_hour'])

    try:
        if not os.path.exists(file_path):
            logger.warning(f"IoT data file '{os.path.basename(file_path)}' not found. Clinic environmental metrics will be unavailable.")
            return default_empty_iot_df

        df = pd.read_csv(file_path)

        if 'timestamp' not in df.columns:
            logger.error("IoT data missing 'timestamp' column. This data is largely unusable without timestamps.")
            st.error("IoT data is missing the critical 'timestamp' column.")
            return default_empty_iot_df # Return with schema
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp is invalid

        required_iot_cols = ['clinic_id', 'room_name', 'avg_co2_ppm'] # Minimal for basic functionality
        missing_iot_cols = [col for col in required_iot_cols if col not in df.columns]
        if missing_iot_cols:
            logger.warning(f"IoT data missing some expected columns: {missing_iot_cols}. Environmental metrics may be incomplete.")

        # Zone ID handling: Prefer direct 'zone_id', fallback to map, then to 'Unknown'
        if 'zone_id' not in df.columns or df['zone_id'].isnull().all():
            if 'clinic_id' in df.columns:
                # This mapping should ideally come from a config file or separate data source
                clinic_to_zone_map_example = { 'C01': 'ZoneA', 'C02': 'ZoneB', 'C03': 'ZoneC', 'C04': 'ZoneD', 'C05':'ZoneE', 'C06':'ZoneF' }
                df['zone_id'] = df['clinic_id'].astype(str).map(clinic_to_zone_map_example).fillna('UnknownZoneByClinic')
                logger.info("Derived 'zone_id' for IoT data from 'clinic_id' using a predefined map.")
            else:
                logger.warning("IoT data missing 'zone_id' and 'clinic_id'. Setting 'zone_id' to 'UnknownZoneDirect'.")
                df['zone_id'] = 'UnknownZoneDirect'
        df['zone_id'] = df['zone_id'].astype(str).fillna('UnknownZoneFill')


        numeric_iot_cols = [
            'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index',
            'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
            'waiting_room_occupancy', 'patient_throughput_per_hour',
            'sanitizer_dispenses_per_hour'
        ]
        for col in numeric_iot_cols:
            if col not in df.columns:
                logger.debug(f"Numeric IoT column '{col}' not found in IoT CSV. Creating as NaN-filled column.")
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure string columns are consistently string type
        string_iot_cols = ['clinic_id', 'room_name']
        for col in string_iot_cols:
            if col in df.columns:
                 df[col] = df[col].astype(str).fillna('Unknown')
                 df.loc[df[col].isin(['', 'nan', 'None']), col] = 'Unknown'
            else: df[col] = 'Unknown'


        logger.info(f"Successfully loaded and preprocessed IoT clinic environment data ({len(df)} rows).")
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"IoT data file '{os.path.basename(file_path)}' is empty.")
        return default_empty_iot_df
    except Exception as e:
        logger.error(f"Error loading IoT clinic environment data: {e}", exc_info=True)
        st.error(f"An error occurred while loading IoT clinic environment data: {e}")
        return default_empty_iot_df


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={
    gpd.GeoDataFrame: hash_geodataframe,
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None
})
def enrich_zone_geodata_with_health_aggregates(zone_gdf_base, health_df_input, iot_df_input=None):
    # Define default empty GDF schema for error cases
    default_enriched_cols = [
        'zone_id', 'name', 'geometry', 'population', 'avg_risk_score', 'active_tb_cases',
        'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'anemia_cases', 'sti_cases',
        'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score',
        'chw_visits_in_zone', 'avg_daily_steps_zone', 'avg_spo2_zone', 'zone_avg_co2', 'num_clinics', 'socio_economic_index'
    ]
    empty_enriched_gdf = gpd.GeoDataFrame(columns=default_enriched_cols, crs=app_config.DEFAULT_CRS)

    if zone_gdf_base is None or zone_gdf_base.empty:
        logger.warning("Zone GeoDataFrame (zone_gdf_base) is empty or None for enrichment. Returning an empty GeoDataFrame with default schema.")
        return empty_enriched_gdf

    enriched_gdf = zone_gdf_base.copy()
    if 'zone_id' not in enriched_gdf.columns:
        logger.error("CRITICAL: 'zone_id' missing in input zone_gdf_base for enrichment. Cannot proceed with meaningful enrichment.")
        return enriched_gdf # Or return empty_enriched_gdf
    enriched_gdf['zone_id'] = enriched_gdf['zone_id'].astype(str)

    # Initialize health aggregate columns to prevent missing columns later
    health_agg_cols_defaults = {
        'avg_risk_score': np.nan, 'active_tb_cases': 0, 'active_malaria_cases': 0,
        'hiv_positive_cases': 0, 'pneumonia_cases': 0, 'anemia_cases': 0, 'sti_cases': 0,
        'chw_visits_in_zone': 0, 'avg_daily_steps_zone': np.nan, 'avg_spo2_zone': np.nan,
        'avg_skin_temp_zone': np.nan, 'total_falls_detected_zone': 0, 'hpv_screenings_done': 0,
        # Add any other aggregates from health_df
    }
    for col, default_val in health_agg_cols_defaults.items():
        enriched_gdf[col] = default_val

    if health_df_input is not None and not health_df_input.empty and 'zone_id' in health_df_input.columns:
        health_df = health_df_input.copy()
        health_df['zone_id'] = health_df['zone_id'].astype(str)

        def count_unique_condition(group, conditions_list):
            return group[group['condition'].isin(conditions_list)]['patient_id'].nunique()

        zone_health_summary = health_df.groupby('zone_id').agg(
            avg_risk_score=('ai_risk_score', 'mean'),
            active_tb_cases=('condition', lambda x: count_unique_condition(x, ['TB'])),
            active_malaria_cases=('condition', lambda x: count_unique_condition(x, ['Malaria'])),
            hiv_positive_cases=('condition', lambda x: count_unique_condition(x, ['HIV-Positive'])),
            pneumonia_cases=('condition', lambda x: count_unique_condition(x, ['Pneumonia'])),
            anemia_cases=('condition', lambda x: count_unique_condition(x, ['Anemia'])),
            sti_cases=('condition', lambda x: x[x.str.startswith('STI-', na=False)].nunique()), # unique patients with any STI
            chw_visits_in_zone=('chw_visit', lambda x: pd.to_numeric(x, errors='coerce').sum()),
            avg_daily_steps_zone=('avg_daily_steps', 'mean'),
            avg_spo2_zone=('avg_spo2', 'mean'),
            avg_skin_temp_zone=('max_skin_temp_celsius', 'mean'), # Or 'avg_skin_temp_celsius' if available
            total_falls_detected_zone=('fall_detected_today', lambda x: pd.to_numeric(x, errors='coerce').sum()),
            hpv_screenings_done=('test_type', lambda x: (x == 'PapSmear').sum())
        ).reset_index()
        enriched_gdf = enriched_gdf.merge(zone_health_summary, on='zone_id', how='left', suffixes=('', '_health_agg'))
        # Clean up merged columns, prioritizing aggregated ones and filling NaNs
        for col_original in zone_health_summary.columns:
            if col_original == 'zone_id': continue
            col_agg = f"{col_original}_health_agg"
            if col_agg in enriched_gdf.columns: # If merge created a suffixed col due to pre-existing col from zone_attributes
                enriched_gdf[col_original] = enriched_gdf[col_agg].combine_first(enriched_gdf[col_original])
                enriched_gdf.drop(columns=[col_agg], inplace=True, errors='ignore')
        # Fill NaNs resulting from left merge (zones with no health data) using the defaults
        for col, default_val in health_agg_cols_defaults.items():
            if col in enriched_gdf.columns:
                 enriched_gdf[col] = enriched_gdf[col].fillna(default_val)


    # Initialize IoT aggregate columns
    iot_agg_cols_defaults = {'zone_avg_co2': np.nan, 'zone_max_co2': np.nan, 'zone_avg_temp': np.nan, 'zone_avg_pm25':np.nan, 'zone_avg_occupancy': np.nan}
    for col, default_val in iot_agg_cols_defaults.items():
        enriched_gdf[col] = default_val

    if iot_df_input is not None and not iot_df_input.empty and 'zone_id' in iot_df_input.columns:
        iot_df = iot_df_input.copy()
        iot_df['zone_id'] = iot_df['zone_id'].astype(str)
        # Consider only latest reading per room per zone for certain IoT avgs if makes sense
        latest_iot_readings_per_room = iot_df.sort_values('timestamp').drop_duplicates(subset=['zone_id', 'clinic_id', 'room_name'], keep='last')

        iot_zone_summary = latest_iot_readings_per_room.groupby('zone_id').agg(
            zone_avg_co2=('avg_co2_ppm', 'mean'),
            zone_max_co2=('max_co2_ppm', 'max'), # Max of latest room maxes
            zone_avg_temp=('avg_temp_celsius', 'mean'),
            zone_avg_pm25=('avg_pm25', 'mean'),
            zone_avg_occupancy=('waiting_room_occupancy', 'mean')
        ).reset_index()
        enriched_gdf = enriched_gdf.merge(iot_zone_summary, on='zone_id', how='left', suffixes=('', '_iot_agg'))
        for col_original in iot_zone_summary.columns:
            if col_original == 'zone_id': continue
            col_agg = f"{col_original}_iot_agg"
            if col_agg in enriched_gdf.columns:
                enriched_gdf[col_original] = enriched_gdf[col_agg].combine_first(enriched_gdf[col_original])
                enriched_gdf.drop(columns=[col_agg], inplace=True, errors='ignore')
        for col, default_val in iot_agg_cols_defaults.items():
            if col in enriched_gdf.columns:
                enriched_gdf[col] = enriched_gdf[col].fillna(default_val)


    # Calculate composite metrics (Prevalence, Facility Coverage)
    key_infection_cols = ['active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'sti_cases']
    # Ensure these columns exist and are numeric (fillna with 0)
    for col in key_infection_cols:
        if col not in enriched_gdf.columns: enriched_gdf[col] = 0
        else: enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce').fillna(0)

    enriched_gdf['total_active_key_infections'] = enriched_gdf[key_infection_cols].sum(axis=1)
    enriched_gdf['population'] = pd.to_numeric(enriched_gdf.get('population', 0), errors='coerce').fillna(0)
    enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(
        lambda row: (row['total_active_key_infections'] / row['population']) * 1000 if row['population'] > 0 else 0.0, axis=1
    ).fillna(0.0)

    # Facility Coverage Score (improved logic)
    if 'avg_travel_time_clinic_min' in enriched_gdf.columns and 'num_clinics' in enriched_gdf.columns:
        # Normalize travel time (lower is better)
        min_travel = enriched_gdf['avg_travel_time_clinic_min'].min(); max_travel = enriched_gdf['avg_travel_time_clinic_min'].max()
        if max_travel > min_travel:
            enriched_gdf['travel_score'] = 100 * (1 - (enriched_gdf['avg_travel_time_clinic_min'].fillna(max_travel) - min_travel) / (max_travel - min_travel))
        elif enriched_gdf['avg_travel_time_clinic_min'].notna().any(): enriched_gdf['travel_score'] = 50.0 # All same or single zone
        else: enriched_gdf['travel_score'] = 0.0

        # Normalize clinic density (clinics per 1k population, higher is better)
        enriched_gdf['clinics_per_1k_pop'] = enriched_gdf.apply(lambda r: (r['num_clinics']/r['population'])*1000 if r['population']>0 else 0, axis=1)
        min_density = enriched_gdf['clinics_per_1k_pop'].min(); max_density = enriched_gdf['clinics_per_1k_pop'].max()
        if max_density > min_density:
            enriched_gdf['clinic_density_score'] = 100 * (enriched_gdf['clinics_per_1k_pop'].fillna(min_density) - min_density) / (max_density - min_density)
        elif enriched_gdf['clinics_per_1k_pop'].notna().any(): enriched_gdf['clinic_density_score'] = 50.0
        else: enriched_gdf['clinic_density_score'] = 0.0
        
        enriched_gdf['facility_coverage_score'] = (enriched_gdf['travel_score'].fillna(0) * 0.5 + enriched_gdf['clinic_density_score'].fillna(0) * 0.5)
    else:
        enriched_gdf['facility_coverage_score'] = 0.0 # Default if columns missing

    # Final check for all numeric aggregate columns, ensure they are float and NaNs are 0 for calculations
    final_numeric_cols_to_check = list(health_agg_cols_defaults.keys()) + list(iot_agg_cols_defaults.keys()) + \
                                  ['total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score',
                                   'population', 'num_clinics', 'socio_economic_index'] # From zone_attributes too
    for col in final_numeric_cols_to_check:
        if col in enriched_gdf.columns:
            enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce').fillna(0.0)
        elif col not in ['geometry', 'name', 'zone_id']: # Ensure these are created if somehow missing and critical
            enriched_gdf[col] = 0.0


    logger.info("Zone geodata successfully enriched with health and IoT aggregates.")
    return enriched_gdf


# --- KPI Calculation Functions ---
@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_overall_kpis(df_health_records, date_filter_end=None):
    # This function computes KPIs based on the state of df_health_records up to date_filter_end
    # For 'active' cases or 'pending' items, it usually considers the latest status of each patient/item.
    default_kpis_overall = {
        "total_patients": 0, "avg_patient_risk": 0.0, "active_tb_cases_current": 0,
        "malaria_rdt_positive_rate_period": 0.0, "hiv_newly_diagnosed_period": 0,
        "pending_critical_referrals_current": 0, "avg_test_turnaround_period": 0.0,
        "critical_supply_shortages_current":0, "anemia_prevalence_women_period": 0.0
    }
    if df_health_records is None or df_health_records.empty:
        logger.warning("Overall KPIs: Input health_records DataFrame is empty or None.")
        return default_kpis_overall

    df = df_health_records.copy()
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.error("Overall KPIs: 'date' column missing or not datetime in health_records.")
        return default_kpis_overall
    
    df.dropna(subset=['date'], inplace=True) # Crucial: remove records with no date
    if df.empty:
        logger.warning("Overall KPIs: DataFrame became empty after dropping records with no date.")
        return default_kpis_overall

    # Determine the end date for calculations (either provided or latest in data)
    current_data_snapshot_date = pd.to_datetime(date_filter_end if date_filter_end else df['date'].max()).normalize()
    
    # Filter data up to the snapshot date for "current state" KPIs
    df_upto_snapshot = df[df['date'] <= current_data_snapshot_date].copy()
    if df_upto_snapshot.empty:
        logger.info(f"Overall KPIs: No data on or before {current_data_snapshot_date}. Returning defaults.")
        return default_kpis_overall

    # --- KPIs based on latest status up to snapshot_date ---
    latest_patient_records = df_upto_snapshot.sort_values('date').drop_duplicates(subset=['patient_id'], keep='last')
    
    total_patients = latest_patient_records['patient_id'].nunique()
    avg_patient_risk = latest_patient_records['ai_risk_score'].mean() if 'ai_risk_score' in latest_patient_records else 0.0
    
    active_tb_cases_current = latest_patient_records[
        (latest_patient_records['condition'] == 'TB') &
        (latest_patient_records.get('referral_status', 'Unknown') != 'Completed') # Simplified: Not 'Completed' means still active
    ]['patient_id'].nunique()

    pending_critical_referrals_current = latest_patient_records[
        (latest_patient_records.get('referral_status', 'Unknown') == 'Pending') &
        (latest_patient_records['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS))
    ]['patient_id'].nunique()

    latest_item_stock_levels = df_upto_snapshot.sort_values('date').drop_duplicates(subset=['item'], keep='last')
    latest_item_stock_levels['days_of_supply'] = latest_item_stock_levels.apply(
        lambda r: (r['stock_on_hand'] / r['consumption_rate_per_day']) if pd.notna(r['consumption_rate_per_day']) and r['consumption_rate_per_day'] > 0 and pd.notna(r['stock_on_hand']) else np.nan, axis=1
    )
    critical_supply_shortages_current = latest_item_stock_levels[
        (latest_item_stock_levels['days_of_supply'].notna()) &
        (latest_item_stock_levels['days_of_supply'] <= app_config.CRITICAL_SUPPLY_DAYS) &
        (latest_item_stock_levels['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False))
    ]['item'].nunique()

    # --- KPIs based on activity within a defined period (e.g., last 30 days from snapshot_date) ---
    period_start_date = current_data_snapshot_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1)
    df_period = df_upto_snapshot[df_upto_snapshot['date'] >= period_start_date].copy()

    if not df_period.empty:
        malaria_tests_period = df_period[df_period['test_type'].isin(['RDT-Malaria', 'Microscopy-Malaria'])]
        malaria_pos_period = malaria_tests_period[malaria_tests_period['test_result'] == 'Positive'].shape[0]
        malaria_conclusive_period = malaria_tests_period[~malaria_tests_period['test_result'].isin(['Pending', 'N/A', 'Unknown'])].shape[0]
        malaria_rdt_positive_rate_period = (malaria_pos_period / malaria_conclusive_period) * 100 if malaria_conclusive_period > 0 else 0.0

        hiv_newly_diagnosed_period = df_period[
            (df_period['condition'] == 'HIV-Positive') &
            (df_period['test_result'] == 'Positive') & # Assuming 'Positive' on a test_type related to HIV implies new diagnosis in this period
            (df_period['test_type'].str.contains("HIV", case=False, na=False))
        ]['patient_id'].nunique() # Unique patients diagnosed in period

        completed_tests_period = df_period[
            df_period['test_turnaround_days'].notna() &
            (~df_period['test_result'].isin(['Pending', 'N/A', 'Unknown'])) # Conclusive tests
        ]
        avg_test_turnaround_period = completed_tests_period['test_turnaround_days'].mean() if not completed_tests_period.empty else 0.0

        # Anemia prevalence among women (example: assuming gender and age are available)
        # This needs a clear definition of "women of reproductive age"
        women_tested_anemia = df_period[
            (df_period.get('gender','Unknown') == 'Female') & # Requires gender data
            (df_period.get('age', 0).between(15,49)) &       # Requires age data
            (df_period['test_type'] == 'Hemoglobin Test') &
            (~df_period['test_result'].isin(['Pending', 'N/A', 'Unknown']))
        ]
        anemia_low_hb = women_tested_anemia[women_tested_anemia['test_result'] == 'Low'].shape[0]
        anemia_prevalence_women_period = (anemia_low_hb / women_tested_anemia.shape[0]) * 100 if not women_tested_anemia.empty else 0.0
    else: # No data in the defined period
        malaria_rdt_positive_rate_period = 0.0
        hiv_newly_diagnosed_period = 0
        avg_test_turnaround_period = 0.0
        anemia_prevalence_women_period = 0.0


    return {
        "total_patients": total_patients,
        "avg_patient_risk": avg_patient_risk if pd.notna(avg_patient_risk) else 0.0,
        "active_tb_cases_current": active_tb_cases_current,
        "malaria_rdt_positive_rate_period": malaria_rdt_positive_rate_period,
        "hiv_newly_diagnosed_period": hiv_newly_diagnosed_period,
        "pending_critical_referrals_current": pending_critical_referrals_current,
        "avg_test_turnaround_period": avg_test_turnaround_period if pd.notna(avg_test_turnaround_period) else 0.0,
        "critical_supply_shortages_current": critical_supply_shortages_current,
        "anemia_prevalence_women_period": anemia_prevalence_women_period
    }


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_chw_summary(df_chw_day_view): # df_chw_day_view is already filtered for a specific day
    default_summary = {
        "visits_today": 0, "tb_contacts_to_trace_today": 0, "sti_symptomatic_referrals_today": 0,
        "avg_patient_risk_visited_today":0.0, "patients_low_spo2_visited_today": 0,
        "patients_fever_visited_today": 0, "avg_patient_steps_visited_today":0.0,
        "high_risk_followups_today":0
    }
    if df_chw_day_view is None or df_chw_day_view.empty:
        logger.info("CHW Summary: Input DataFrame for the day is empty or None.")
        return default_summary

    df = df_chw_day_view.copy()
    # Ensure relevant columns are numeric and correctly typed
    numeric_cols_chw = ['chw_visit', 'tb_contact_traced', 'ai_risk_score', 'avg_spo2', 'min_spo2_pct', 'max_skin_temp_celsius', 'avg_daily_steps', 'fall_detected_today']
    for col in numeric_cols_chw:
        df[col] = pd.to_numeric(df.get(col), errors='coerce') # NaNs handled by specific calculations

    visits_today = int(df.get('chw_visit', pd.Series(dtype=float)).sum())

    tb_contacts_to_trace_today = df[
        (df['condition'] == 'TB') &
        (df.get('tb_contact_traced', pd.Series(dtype=float)).fillna(1) == 0) # if missing, assume traced (safer)
    ].shape[0]

    sti_symptomatic_referrals_today = df[
        (df.get('condition', pd.Series(dtype=str)).str.startswith("STI-", na=False)) &
        (df.get('referral_status', 'Unknown') == 'Pending')
    ].shape[0]

    patients_visited_df = df[df.get('chw_visit', pd.Series(dtype=float)) == 1] # Patients CHW interacted with
    
    avg_patient_risk_visited_today = patients_visited_df['ai_risk_score'].mean() if not patients_visited_df.empty and 'ai_risk_score' in patients_visited_df else 0.0
    
    patients_low_spo2_visited_today = 0; patients_fever_visited_today = 0; avg_patient_steps_visited_today = 0.0
    if not patients_visited_df.empty:
        if 'min_spo2_pct' in patients_visited_df: # Prefer min_spo2_pct if available for alerts
            patients_low_spo2_visited_today = patients_visited_df[patients_visited_df['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT].shape[0]
        elif 'avg_spo2' in patients_visited_df: # Fallback to avg_spo2
            patients_low_spo2_visited_today = patients_visited_df[patients_visited_df['avg_spo2'] < app_config.SPO2_LOW_THRESHOLD_PCT].shape[0]

        if 'max_skin_temp_celsius' in patients_visited_df:
            patients_fever_visited_today = patients_visited_df[patients_visited_df['max_skin_temp_celsius'] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C].shape[0]
        
        if 'avg_daily_steps' in patients_visited_df:
            avg_patient_steps_visited_today = patients_visited_df['avg_daily_steps'].mean()
            
    # High-risk patients recorded/updated today that need CHW attention
    high_risk_followups_today = df[
        (df.get('ai_risk_score', 0) >= app_config.RISK_THRESHOLDS['chw_alert_high']) | # High AI risk OR
        (df.get('min_spo2_pct', 100) < app_config.SPO2_CRITICAL_THRESHOLD_PCT) | # Critical SpO2 OR
        (df.get('fall_detected_today', 0) > 0) # Fall detected
    ].shape[0]


    return {
        "visits_today": visits_today,
        "tb_contacts_to_trace_today": tb_contacts_to_trace_today,
        "sti_symptomatic_referrals_today": sti_symptomatic_referrals_today,
        "avg_patient_risk_visited_today": avg_patient_risk_visited_today if pd.notna(avg_patient_risk_visited_today) else 0.0,
        "patients_low_spo2_visited_today": patients_low_spo2_visited_today,
        "patients_fever_visited_today": patients_fever_visited_today,
        "avg_patient_steps_visited_today": avg_patient_steps_visited_today if pd.notna(avg_patient_steps_visited_today) else 0.0,
        "high_risk_followups_today": high_risk_followups_today
    }

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_chw(df_chw_day_view, risk_threshold_moderate=None): # df_chw_day_view is specific to the day
    if df_chw_day_view is None or df_chw_day_view.empty:
        return pd.DataFrame()

    df = df_chw_day_view.copy()
    # Ensure columns for alerts logic, with type safety
    alert_logic_cols = {
        'ai_risk_score': 'float', 'min_spo2_pct': 'float', 'max_skin_temp_celsius': 'float',
        'fall_detected_today': 'int', 'tb_contact_traced': 'int', 'condition': 'str',
        'referral_status': 'str', 'patient_id': 'str', 'zone_id': 'str', 'date': 'datetime'
    }
    for col, dtype in alert_logic_cols.items():
        if col not in df.columns:
            df[col] = np.nan if dtype == 'float' else (0 if dtype == 'int' else 'Unknown')
        if dtype == 'float': df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == 'int': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        elif dtype == 'datetime' : df[col] = pd.to_datetime(df.get(col), errors='coerce') # Already handled in load
        else: df[col] = df[col].astype(str).fillna('Unknown')


    risk_mod = risk_threshold_moderate if risk_threshold_moderate is not None else app_config.RISK_THRESHOLDS['chw_alert_moderate']
    risk_high = app_config.RISK_THRESHOLDS['chw_alert_high']
    spo2_crit = app_config.SPO2_CRITICAL_THRESHOLD_PCT
    spo2_low = app_config.SPO2_LOW_THRESHOLD_PCT
    fever_thresh = app_config.SKIN_TEMP_FEVER_THRESHOLD_C

    s_false = pd.Series([False]*len(df), index=df.index) # Default boolean Series

    # Alert Conditions (prioritized by potential severity)
    cond_fall = df.get('fall_detected_today', s_false.copy()) > 0
    cond_critical_spo2 = df.get('min_spo2_pct', s_false.copy()) < spo2_crit
    cond_fever = df.get('max_skin_temp_celsius', s_false.copy()) >= fever_thresh
    cond_high_risk_score = df.get('ai_risk_score', s_false.copy()) >= risk_high
    
    cond_low_spo2_mod = (df.get('min_spo2_pct', s_false.copy()) >= spo2_crit) & (df.get('min_spo2_pct', s_false.copy()) < spo2_low)
    cond_mod_risk_score = (df.get('ai_risk_score', s_false.copy()) >= risk_mod) & (df.get('ai_risk_score', s_false.copy()) < risk_high)
    cond_tb_trace_needed = (df.get('condition', s_false.copy()) == 'TB') & (df.get('tb_contact_traced', s_false.copy()) == 0)
    cond_key_referral_pending = (df.get('referral_status', s_false.copy()) == 'Pending') & (df.get('condition', s_false.copy()).isin(app_config.KEY_CONDITIONS_FOR_TRENDS))
    
    # Combine all conditions that warrant an alert for the CHW
    alertable_conditions_mask = (cond_fall | cond_critical_spo2 | cond_fever | cond_high_risk_score |
                                 cond_low_spo2_mod | cond_mod_risk_score | cond_tb_trace_needed | cond_key_referral_pending)
    alerts_df = df[alertable_conditions_mask].copy()

    if alerts_df.empty: return pd.DataFrame()

    def determine_chw_alert_reason_and_priority(row):
        reasons = []; priority = 0 # Higher priority value is more urgent
        # Most critical alerts first
        if row.get('fall_detected_today', 0) > 0: reasons.append(f"Fall ({int(row['fall_detected_today'])})"); priority += 100
        if pd.notna(row.get('min_spo2_pct')):
            if row['min_spo2_pct'] < spo2_crit: reasons.append(f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)"); priority += 90
            elif row['min_spo2_pct'] < spo2_low: reasons.append(f"Low SpO2 ({row['min_spo2_pct']:.0f}%)"); priority += 50
        if pd.notna(row.get('max_skin_temp_celsius')) and row['max_skin_temp_celsius'] >= fever_thresh:
            reasons.append(f"Fever ({row['max_skin_temp_celsius']:.1f}°C)"); priority += 70
        
        # High Risk based on AI score
        if pd.notna(row.get('ai_risk_score')):
            if row['ai_risk_score'] >= risk_high: reasons.append(f"High Risk ({row['ai_risk_score']:.0f})"); priority += 80
            elif row['ai_risk_score'] >= risk_mod: reasons.append(f"Mod. Risk ({row['ai_risk_score']:.0f})"); priority += 40
        
        # Disease-specific tasks
        if row.get('condition') == 'TB' and row.get('tb_contact_traced', 1) == 0: reasons.append("TB Contact Trace"); priority += 60
        if row.get('referral_status') == 'Pending' and row.get('condition') in app_config.KEY_CONDITIONS_FOR_TRENDS:
            reasons.append(f"Referral ({row.get('condition')})"); priority += 30
        
        return "; ".join(reasons) if reasons else "Review Case", priority

    alerts_df[['alert_reason', 'priority_score']] = alerts_df.apply(
        lambda row: pd.Series(determine_chw_alert_reason_and_priority(row)), axis=1
    )
    
    alerts_df.sort_values(by=['priority_score', 'ai_risk_score'], ascending=[False, False], inplace=True)
    # Ensure one alert per patient, choosing the one with highest priority from this day's records
    alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)

    # Select and order columns for the final output DataFrame
    output_cols_chw_alerts = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason',
                              'referral_status', 'min_spo2_pct', 'max_skin_temp_celsius',
                              'fall_detected_today', 'priority_score', 'date']
    final_alert_cols = [col for col in output_cols_chw_alerts if col in alerts_df.columns]
    return alerts_df[final_alert_cols]


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_summary(df_clinic_period_view): # df_clinic_period_view is filtered for a date range
    default_summary = {
        "tb_sputum_positivity": 0.0, "malaria_positivity": 0.0, "sti_critical_tests_pending": 0,
        "hiv_tests_conclusive_period": 0, "key_drug_stockouts": 0,
        "avg_test_turnaround_all_tests":0.0, "hpv_screening_coverage_proxy":0.0,
        "avg_patient_risk_clinic":0.0 # New: Avg risk of patients visiting this clinic in period
    }
    if df_clinic_period_view is None or df_clinic_period_view.empty: return default_summary

    df = df_clinic_period_view.copy()
    # Standardize string columns for consistent logic
    str_cols = ['test_type', 'test_result', 'item', 'condition', 'patient_id']
    num_cols = ['stock_on_hand', 'consumption_rate_per_day', 'test_turnaround_days', 'ai_risk_score']
    for col in str_cols: df[col] = df.get(col, pd.Series(dtype='str')).astype(str).fillna('Unknown').replace(['', 'nan', 'None'], 'Unknown')
    for col in num_cols: df[col] = pd.to_numeric(df.get(col), errors='coerce') # NaNs handled by calcs

    # TB Sputum/GeneXpert Positivity
    tb_tests = df[df['test_type'].str.contains("Sputum|GeneXpert", case=False, na=False)]
    tb_positive = tb_tests[tb_tests['test_result'] == 'Positive']['patient_id'].nunique() # Unique patients positive
    tb_total_conclusive = tb_tests[~tb_tests['test_result'].isin(['Pending', 'N/A', 'Unknown'])]['patient_id'].nunique() # Unique patients tested
    tb_sputum_positivity = (tb_positive / tb_total_conclusive) * 100 if tb_total_conclusive > 0 else 0.0

    # Malaria Positivity (RDT or Microscopy)
    malaria_tests = df[df['test_type'].str.contains("RDT-Malaria|Microscopy-Malaria", case=False, na=False)]
    malaria_positive = malaria_tests[malaria_tests['test_result'] == 'Positive']['patient_id'].nunique()
    malaria_total_conclusive = malaria_tests[~malaria_tests['test_result'].isin(['Pending', 'N/A', 'Unknown'])]['patient_id'].nunique()
    malaria_positivity = (malaria_positive / malaria_total_conclusive) * 100 if malaria_total_conclusive > 0 else 0.0

    # Critical STI Tests Pending
    sti_critical_tests_pending = df[
        (df['test_type'].isin(app_config.CRITICAL_TESTS_PENDING)) &
        (df['condition'].str.startswith("STI-", na=False)) &
        (df['test_result'] == 'Pending')
    ]['patient_id'].nunique() # Number of unique patients with pending critical STI tests

    # HIV Tests Conducted (conclusive results)
    hiv_tests_conclusive_period = df[
        df['test_type'].str.contains("HIV", case=False, na=False) &
        (~df['test_result'].isin(['Pending','N/A','Unknown']))
    ]['patient_id'].nunique() # Unique patients with conclusive HIV test results

    # Key Drug Stockouts (Days of Supply < Threshold)
    df['days_of_supply'] = df.apply(
        lambda row: (row['stock_on_hand'] / row['consumption_rate_per_day'])
        if pd.notna(row['stock_on_hand']) and pd.notna(row['consumption_rate_per_day']) and row['consumption_rate_per_day'] > 0
        else np.nan, axis=1
    )
    # Consider latest stock level for each item in the period
    latest_stock_in_period = df.sort_values('date').drop_duplicates(subset=['item'], keep='last')
    key_drug_stockouts = latest_stock_in_period[
        (latest_stock_in_period['days_of_supply'].notna()) &
        (latest_stock_in_period['days_of_supply'] <= app_config.CRITICAL_SUPPLY_DAYS) &
        (latest_stock_in_period['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False))
    ]['item'].nunique()
    
    # Average Test Turnaround Time (Overall for conclusive tests in period)
    conclusive_tests_with_tat_period = df[
        (~df['test_result'].isin(['Pending', 'N/A', 'Unknown'])) &
        (df['test_turnaround_days'].notna())
    ]
    avg_test_turnaround_all_tests = conclusive_tests_with_tat_period['test_turnaround_days'].mean()

    # HPV Screening Proxy (PapSmears per 100 unique female patients aged 21-65 if data available)
    # Simplified: count of PapSmears / unique patients in period (very rough proxy)
    hpv_screenings_done = df[df['test_type'] == 'PapSmear']['patient_id'].nunique() # Unique patients screened
    # Proper denominator needs age/gender from a patient master. Here, using total unique patients as proxy denominator.
    total_unique_patients_in_period = df['patient_id'].nunique()
    hpv_screening_coverage_proxy = (hpv_screenings_done / total_unique_patients_in_period) * 100 if total_unique_patients_in_period > 0 else 0.0
    
    # Average AI risk score of unique patients attending clinic in period
    avg_patient_risk_clinic = df.drop_duplicates(subset=['patient_id'])['ai_risk_score'].mean()


    return {
        "tb_sputum_positivity": tb_sputum_positivity,
        "malaria_positivity": malaria_positivity,
        "sti_critical_tests_pending": sti_critical_tests_pending,
        "hiv_tests_conclusive_period": hiv_tests_conclusive_period,
        "key_drug_stockouts": key_drug_stockouts,
        "avg_test_turnaround_all_tests": avg_test_turnaround_all_tests if pd.notna(avg_test_turnaround_all_tests) else 0.0,
        "hpv_screening_coverage_proxy": hpv_screening_coverage_proxy,
        "avg_patient_risk_clinic": avg_patient_risk_clinic if pd.notna(avg_patient_risk_clinic) else 0.0
    }

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_environmental_summary(df_iot_clinic_period_view): # df_iot_clinic_period_view filtered for date range
    default_summary = {
        "avg_co2_overall": 0.0, "rooms_co2_alert_latest": 0, "avg_pm25_overall": 0.0,
        "rooms_pm25_alert_latest": 0, "avg_occupancy_overall":0.0,
        "high_occupancy_alert_latest": False, "avg_sanitizer_use_hr_overall":0.0,
        "rooms_noise_alert_latest":0
    }
    if df_iot_clinic_period_view is None or df_iot_clinic_period_view.empty: return default_summary

    df = df_iot_clinic_period_view.copy()
    # Ensure numeric types
    iot_numeric_cols = ['avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy',
                        'sanitizer_dispenses_per_hour', 'avg_noise_db', 'max_co2_ppm']
    for col in iot_numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors='coerce')

    # Overall averages for the period for all readings
    avg_co2_overall = df['avg_co2_ppm'].mean()
    avg_pm25_overall = df['avg_pm25'].mean()
    avg_occupancy_overall = df['waiting_room_occupancy'].mean()
    avg_sanitizer_use_hr_overall = df['sanitizer_dispenses_per_hour'].mean()

    # Alerts based on *latest* readings per room within the filtered period
    rooms_co2_alert_latest = 0; rooms_pm25_alert_latest = 0; high_occupancy_alert_latest = False; rooms_noise_alert_latest = 0
    if 'timestamp' in df and 'clinic_id' in df and 'room_name' in df: # Required for 'latest per room' logic
        # If multiple clinics are in the view, this needs to be per clinic-room
        latest_readings_per_room = df.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        if not latest_readings_per_room.empty:
            if 'avg_co2_ppm' in latest_readings_per_room:
                rooms_co2_alert_latest = latest_readings_per_room[latest_readings_per_room['avg_co2_ppm'] > app_config.CO2_LEVEL_ALERT_PPM].shape[0]
            if 'avg_pm25' in latest_readings_per_room:
                rooms_pm25_alert_latest = latest_readings_per_room[latest_readings_per_room['avg_pm25'] > app_config.PM25_ALERT_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_readings_per_room: # Alert if ANY room's latest reading is over target
                high_occupancy_alert_latest = (latest_readings_per_room['waiting_room_occupancy'] > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            if 'avg_noise_db' in latest_readings_per_room:
                rooms_noise_alert_latest = latest_readings_per_room[latest_readings_per_room['avg_noise_db'] > app_config.NOISE_LEVEL_ALERT_DB].shape[0]

    return {
        "avg_co2_overall": avg_co2_overall if pd.notna(avg_co2_overall) else 0.0,
        "rooms_co2_alert_latest": rooms_co2_alert_latest,
        "avg_pm25_overall": avg_pm25_overall if pd.notna(avg_pm25_overall) else 0.0,
        "rooms_pm25_alert_latest": rooms_pm25_alert_latest,
        "avg_occupancy_overall": avg_occupancy_overall if pd.notna(avg_occupancy_overall) else 0.0,
        "high_occupancy_alert_latest": bool(high_occupancy_alert_latest),
        "avg_sanitizer_use_hr_overall": avg_sanitizer_use_hr_overall if pd.notna(avg_sanitizer_use_hr_overall) else 0.0,
        "rooms_noise_alert_latest": rooms_noise_alert_latest
    }


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_clinic(df_clinic_period_view, risk_threshold_moderate=None):
    # df_clinic_period_view is already filtered for a specific date range
    if df_clinic_period_view is None or df_clinic_period_view.empty: return pd.DataFrame()

    df = df_clinic_period_view.copy()
    # Date columns must be valid for time-based logic
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']): return pd.DataFrame()
    df.dropna(subset=['date'], inplace=True)
    df['test_date'] = pd.to_datetime(df.get('test_date'), errors='coerce') # Ensure test_date is datetime

    df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score'), errors='coerce')
    str_cols_clinic_alert = ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id', 'test_type']
    for col in str_cols_clinic_alert: df[col] = df.get(col, pd.Series(dtype='str')).astype(str).fillna('Unknown').replace(['', 'nan', 'None'], 'Unknown')
    df['hiv_viral_load'] = pd.to_numeric(df.get('hiv_viral_load'), errors='coerce')


    risk_mod_clinic = risk_threshold_moderate if risk_threshold_moderate is not None else app_config.RISK_THRESHOLDS['moderate']
    risk_high_clinic = app_config.RISK_THRESHOLDS['high']
    
    # Use the latest date from the input df_clinic_period_view as the reference for "recent" or "overdue"
    current_snapshot_date_clinic = df['date'].max().normalize()
    recent_positive_lookback_days = 7
    overdue_test_lookback_days = 10 # Test taken more than 10 days ago and still pending

    s_false = pd.Series([False]*len(df), index=df.index) # Default boolean Series

    # Define alert conditions based on the latest record for each patient within the period
    latest_records_in_period = df.sort_values('date').drop_duplicates(subset=['patient_id', 'condition'], keep='last') # Latest status for each condition

    cond_clinic_high_risk = latest_records_in_period.get('ai_risk_score', s_false.copy()) >= risk_high_clinic
    
    cond_clinic_recent_critical_positive = (
        (latest_records_in_period.get('test_result', s_false.copy()) == 'Positive') &
        (latest_records_in_period.get('condition', s_false.copy()).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)) &
        (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)).notna()) &
        (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)) >= (current_snapshot_date_clinic - pd.Timedelta(days=recent_positive_lookback_days)))
    )
    cond_clinic_overdue_critical_test = (
        (latest_records_in_period.get('test_result', s_false.copy()) == 'Pending') &
        (latest_records_in_period.get('test_type', s_false.copy()).isin(app_config.CRITICAL_TESTS_PENDING)) &
        (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)).notna()) &
        (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)) < (current_snapshot_date_clinic - pd.Timedelta(days=overdue_test_lookback_days)))
    )
    cond_clinic_hiv_high_vl = (
        (latest_records_in_period.get('condition', s_false.copy()) == 'HIV-Positive') &
        (latest_records_in_period.get('hiv_viral_load', pd.Series(dtype=float)).notna()) &
        (latest_records_in_period.get('hiv_viral_load', pd.Series(dtype=float)) > 1000) # Example high VL threshold
    )
    
    # Apply these conditions to the 'latest_records_in_period' DataFrame
    alert_mask_clinic = (cond_clinic_high_risk | cond_clinic_recent_critical_positive |
                         cond_clinic_overdue_critical_test | cond_clinic_hiv_high_vl)
    
    alerts_df_clinic = latest_records_in_period[alert_mask_clinic].copy()

    if alerts_df_clinic.empty: return pd.DataFrame()

    def determine_clinic_alert_reason_and_priority(row):
        reasons = []; priority_score = 0
        # Determine reasons based on the flags (could use the original conditions directly on the row)
        # HIV High VL (most specific action)
        if row.get('condition') == 'HIV-Positive' and pd.notna(row.get('hiv_viral_load')) and row['hiv_viral_load'] > 1000:
            reasons.append(f"High HIV VL ({row['hiv_viral_load']:.0f})"); priority_score += 100
        # High Risk Score
        if pd.notna(row.get('ai_risk_score')) and row['ai_risk_score'] >= risk_high_clinic:
            reasons.append(f"High Risk ({row['ai_risk_score']:.0f})"); priority_score += row['ai_risk_score'] # Add risk score to priority
        # Recent Critical Positive
        if (row.get('test_result') == 'Positive' and row.get('condition') in app_config.KEY_CONDITIONS_FOR_TRENDS and
            pd.notna(row.get('test_date')) and row['test_date'] >= (current_snapshot_date_clinic - pd.Timedelta(days=recent_positive_lookback_days))):
            reasons.append(f"Recent Critical Positive ({row.get('condition')})"); priority_score += 70
        # Overdue Critical Test
        if (row.get('test_result') == 'Pending' and row.get('test_type') in app_config.CRITICAL_TESTS_PENDING and
            pd.notna(row.get('test_date')) and row['test_date'] < (current_snapshot_date_clinic - pd.Timedelta(days=overdue_test_lookback_days))):
            days_pending = (current_snapshot_date_clinic - row['test_date']).days
            reasons.append(f"Overdue Test ({row.get('test_type')}, {days_pending}d)"); priority_score += 60
        
        return "; ".join(reasons) if reasons else "Review Case", priority_score

    alerts_df_clinic[['alert_reason', 'priority_score']] = alerts_df_clinic.apply(
        lambda row: pd.Series(determine_clinic_alert_reason_and_priority(row)), axis=1
    )
    
    alerts_df_clinic.sort_values(by=['priority_score', 'date'], ascending=[False, False], inplace=True)
    # Do not drop duplicates here, as a patient might have multiple alert reasons across different conditions
    # The original df `latest_records_in_period` already took latest per patient-condition.
    
    output_cols_clinic_alerts = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result',
                                 'test_type', 'referral_status', 'alert_reason', 'hiv_viral_load',
                                 'priority_score', 'date']
    final_cols_clinic = [col for col in output_cols_clinic_alerts if col in alerts_df_clinic.columns]
    return alerts_df_clinic[final_cols_clinic]


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_trend_data(df_input, value_col, date_col='date', period='D', agg_func='mean'):
    default_index_name = date_col if df_input is not None and date_col in df_input.columns and df_input[date_col] is not None else 'date'
    empty_series = pd.Series(dtype='float64', name=value_col).rename_axis(default_index_name)

    if df_input is None or df_input.empty:
        logger.debug(f"Trend data: Input DataFrame is None or empty for value_col '{value_col}'.")
        return empty_series
    
    df = df_input.copy() # Work on a copy
    
    if date_col not in df.columns:
        logger.warning(f"Trend data: Date column '{date_col}' not found for value_col '{value_col}'.")
        return empty_series
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True) # Crucial: remove rows where date conversion failed or date is NaT

    # value_col existence check and type conversion
    if value_col not in df.columns:
        if agg_func == 'count': # If agg_func is 'count', value_col presence isn't strictly necessary (can count rows)
            logger.debug(f"Trend data: Value column '{value_col}' not found, but agg_func is 'count'. Will count rows per period.")
        else:
            logger.warning(f"Trend data: Value column '{value_col}' not found for aggregation '{agg_func}'.")
            return empty_series
    elif agg_func in ['mean', 'sum', 'median', 'std']: # Numeric aggregations require numeric value_col
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df.dropna(subset=[value_col], inplace=True) # Remove rows where value is not numeric for these aggs

    if df.empty:
        logger.debug(f"Trend data: DataFrame became empty after date/value processing for '{value_col}'.")
        return empty_series
    
    try:
        grouped = df.set_index(date_col).sort_index() # Set date_col as index for resampling
        if value_col in grouped.columns:
            trend_series = grouped[value_col].resample(period).agg(agg_func)
        elif agg_func == 'count': # If value_col was not present, but we want to count rows.
             trend_series = grouped.resample(period).size()
             trend_series.name = 'count' # Name the series appropriately
        else: # value_col not in grouped columns and agg_func needs it.
            logger.warning(f"Trend data: Value column '{value_col}' not usable after setting index for agg '{agg_func}'.")
            return empty_series

        # Fill NaNs that result from resampling (e.g., periods with no data)
        # For counts/sums, 0 is appropriate. For means/medians, NaN might be better or interpolation.
        fill_value = 0 if agg_func in ['count', 'sum', 'nunique'] else np.nan
        return trend_series.fillna(fill_value)
    
    except Exception as e:
        logger.error(f"Error calculating trend for '{value_col}' on '{date_col}' with agg '{agg_func}': {e}", exc_info=True)
        return empty_series


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_supply_forecast_data(df_health_records_full, forecast_days_out=21): # Default 3 weeks forecast
    cols = ['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci', 'current_stock', 'consumption_rate', 'estimated_stockout_date']
    empty_forecast_df = pd.DataFrame(columns=cols)
    
    if df_health_records_full is None or df_health_records_full.empty:
        logger.warning("Supply Forecast: Input health_records DataFrame is empty or None.")
        return empty_forecast_df
    
    df = df_health_records_full.copy()
    required_cols = ['date', 'item', 'stock_on_hand', 'consumption_rate_per_day']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.warning(f"Supply Forecast: Missing required columns: {missing}. Cannot generate forecast.")
        return empty_forecast_df

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['item'] = df['item'].astype(str).fillna('Unknown').replace(['', 'nan', 'None'], 'Unknown')
    df['stock_on_hand'] = pd.to_numeric(df['stock_on_hand'], errors='coerce')
    df['consumption_rate_per_day'] = pd.to_numeric(df['consumption_rate_per_day'], errors='coerce')

    df.dropna(subset=['date', 'item', 'stock_on_hand'], inplace=True) # Consumption rate handled below
    df = df[~df['item'].isin(['Unknown'])] # Exclude unknown items
    df = df[df['stock_on_hand'] >= 0] # Stock cannot be negative

    if df.empty:
        logger.info("Supply Forecast: No valid data after initial filtering.")
        return empty_forecast_df

    # Get the latest record for each supply item to determine current stock and consumption rate
    latest_supplies = df.sort_values('date').drop_duplicates(subset=['item'], keep='last').copy()
    
    # Estimate consumption rate if missing or zero, but stock exists
    # If consumption_rate is 0 or NaN, but stock_on_hand > 0, this implies either no use or unrecorded use.
    # A robust system would estimate based on historical dispensing or issue alerts for data quality.
    # Simplified: assume a minimal default consumption if rate is 0/NaN and stock exists.
    default_min_consumption = 0.1
    latest_supplies['consumption_rate_per_day'] = latest_supplies.apply(
        lambda row: row['consumption_rate_per_day'] if pd.notna(row['consumption_rate_per_day']) and row['consumption_rate_per_day'] > 0
        else (default_min_consumption if row['stock_on_hand'] > 0 else 0),
        axis=1
    )

    forecast_list = []
    today_for_forecast = pd.Timestamp('today').normalize() # Use current actual date as reference for "days from now"

    for _, row in latest_supplies.iterrows():
        item_name = row['item']
        current_stock_amount = row['stock_on_hand']
        consumption_rate = row['consumption_rate_per_day']
        # last_data_date = row['date'] # Date of last record, forecast starts from "today_for_forecast"

        # Initial point: current days of supply
        days_supply_at_start = current_stock_amount / consumption_rate if consumption_rate > 0 else (np.inf if current_stock_amount > 0 else 0)
        estimated_stockout_date = today_for_forecast + pd.Timedelta(days=days_supply_at_start) if consumption_rate > 0 and current_stock_amount > 0 else pd.NaT
        if days_supply_at_start == np.inf : estimated_stockout_date = pd.NaT # Effectively no stockout

        forecast_list.append({
            'date': today_for_forecast, 'item': item_name,
            'forecast_days': days_supply_at_start,
            'lower_ci': days_supply_at_start, 'upper_ci': days_supply_at_start, # No CI for current day
            'current_stock': current_stock_amount, 'consumption_rate': consumption_rate,
            'estimated_stockout_date': estimated_stockout_date
        })

        if consumption_rate <= 0: # If no consumption, forecast remains flat
            for i in range(1, forecast_days_out + 1):
                forecast_date_iter = today_for_forecast + pd.Timedelta(days=i)
                forecast_list.append({
                    'date': forecast_date_iter, 'item': item_name,
                    'forecast_days': days_supply_at_start, # Stays same (inf or 0)
                    'lower_ci': days_supply_at_start, 'upper_ci': days_supply_at_start,
                    'current_stock': current_stock_amount, 'consumption_rate': consumption_rate,
                    'estimated_stockout_date': estimated_stockout_date
                })
            continue # Next item

        # Daily forecast if consumption > 0
        for i in range(1, forecast_days_out + 1):
            forecast_date_iter = today_for_forecast + pd.Timedelta(days=i)
            forecasted_stock_level = max(0, current_stock_amount - (consumption_rate * i))
            days_of_supply_fc = forecasted_stock_level / consumption_rate if consumption_rate > 0 else (np.inf if forecasted_stock_level > 0 else 0)
            
            # Simple CI: assumes consumption rate can vary by +/- X%
            consumption_ci_factor = 0.25 # e.g., 25% variability
            lower_cons = consumption_rate * (1 - consumption_ci_factor)
            upper_cons = consumption_rate * (1 + consumption_ci_factor)

            stock_at_upper_cons = max(0, current_stock_amount - (upper_cons * i))
            lower_ci_days_fc = stock_at_upper_cons / upper_cons if upper_cons > 0 else (np.inf if stock_at_upper_cons > 0 else 0)
            
            stock_at_lower_cons = max(0, current_stock_amount - (lower_cons * i))
            upper_ci_days_fc = stock_at_lower_cons / lower_cons if lower_cons > 0 else (np.inf if stock_at_lower_cons > 0 else 0)
            
            forecast_list.append({
                'date': forecast_date_iter, 'item': item_name,
                'forecast_days': days_of_supply_fc,
                'lower_ci': lower_ci_days_fc, 'upper_ci': upper_ci_days_fc,
                'current_stock': forecasted_stock_level, # This is the forecasted stock amount on that day
                'consumption_rate': consumption_rate,
                'estimated_stockout_date': estimated_stockout_date # Remains same for the item
            })
            if forecasted_stock_level == 0 and days_of_supply_fc == 0: break # Stop if stock hits zero

    return pd.DataFrame(forecast_list) if forecast_list else empty_forecast_df


@st.cache_data(hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def get_district_summary_kpis(enriched_zone_gdf):
    default_kpis = {
        "avg_population_risk": 0.0, "overall_facility_coverage": 0.0, "zones_high_risk_count": 0,
        "district_tb_burden_total":0, "district_malaria_burden_total":0, "avg_clinic_co2_district":0.0,
        "population_weighted_avg_steps": 0.0, "population_weighted_avg_spo2": 0.0,
        "key_infection_prevalence_district_per_1000": 0.0
    }
    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
        logger.warning("District KPIs: Enriched zone GDF is empty or None.")
        return default_kpis

    gdf = enriched_zone_gdf.copy()
    # Ensure all required columns are numeric and fill NaNs with 0 for calculations
    # These columns are expected to be created by `enrich_zone_geodata_with_health_aggregates`
    kpi_calc_cols = {
        'population': 0.0, 'avg_risk_score': 0.0, 'facility_coverage_score': 0.0,
        'active_tb_cases': 0.0, 'active_malaria_cases': 0.0, 'zone_avg_co2': 0.0,
        'avg_daily_steps_zone': 0.0, 'avg_spo2_zone': 0.0, 'total_active_key_infections':0.0
    }
    for col, default_val in kpi_calc_cols.items():
        if col not in gdf.columns:
            logger.warning(f"District KPIs: Column '{col}' missing in enriched GDF. Calculations may be affected.")
            gdf[col] = default_val # Add with default if missing
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(default_val)

    total_district_population = gdf['population'].sum()
    if total_district_population == 0:
        logger.warning("District KPIs: Total district population is 0. Weighted averages will be 0 or NaN.")
        # Return mostly defaults, but simple sums might still be valid
        district_tb_total = gdf['active_tb_cases'].sum()
        district_mal_total = gdf['active_malaria_cases'].sum()
        return {**default_kpis, "district_tb_burden_total": int(district_tb_total), "district_malaria_burden_total": int(district_mal_total)}


    avg_pop_risk = (gdf['avg_risk_score'] * gdf['population']).sum() / total_district_population
    overall_facility_coverage = (gdf['facility_coverage_score'] * gdf['population']).sum() / total_district_population
    zones_high_risk_count = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]
    
    district_tb_burden_total = gdf['active_tb_cases'].sum()
    district_malaria_burden_total = gdf['active_malaria_cases'].sum()
    # For avg_clinic_co2_district, take a simple average of zonal averages (not population weighted, as it's facility-specific)
    avg_clinic_co2_district = gdf['zone_avg_co2'][gdf['zone_avg_co2'] > 0].mean() if gdf['zone_avg_co2'][gdf['zone_avg_co2'] > 0].notna().any() else 0.0

    pop_weighted_avg_steps = (gdf['avg_daily_steps_zone'] * gdf['population']).sum() / total_district_population
    pop_weighted_avg_spo2 = (gdf['avg_spo2_zone'] * gdf['population']).sum() / total_district_population

    total_key_infections_district = gdf['total_active_key_infections'].sum()
    key_infection_prevalence_district_per_1000 = (total_key_infections_district / total_district_population) * 1000 if total_district_population > 0 else 0.0


    return {
        "avg_population_risk": avg_pop_risk if pd.notna(avg_pop_risk) else 0.0,
        "overall_facility_coverage": overall_facility_coverage if pd.notna(overall_facility_coverage) else 0.0,
        "zones_high_risk_count": int(zones_high_risk_count),
        "district_tb_burden_total": int(district_tb_burden_total),
        "district_malaria_burden_total": int(district_malaria_burden_total),
        "avg_clinic_co2_district": avg_clinic_co2_district if pd.notna(avg_clinic_co2_district) else 0.0,
        "population_weighted_avg_steps": pop_weighted_avg_steps if pd.notna(pop_weighted_avg_steps) else 0.0,
        "population_weighted_avg_spo2": pop_weighted_avg_spo2 if pd.notna(pop_weighted_avg_spo2) else 0.0,
        "key_infection_prevalence_district_per_1000": key_infection_prevalence_district_per_1000
    }
