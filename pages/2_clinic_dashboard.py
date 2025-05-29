# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data,
    get_clinic_summary, get_clinic_environmental_summary,
    get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css(): # pragma: no cover
    css_file = getattr(app_config, 'STYLE_CSS', 'assets/style.css') # Fallback if not in app_config
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {css_file}. Default styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS if hasattr(app_config, 'CACHE_TTL_SECONDS') else 3600)
def get_clinic_page_data_extended():
    logger.info("Attempting to load extended data for Clinic Dashboard...")
    health_df_loaded = load_health_records() # This function MUST be robust
    iot_df_loaded = load_iot_clinic_environment_data() # This function MUST be robust

    # Log details of loaded data
    if health_df_loaded is not None and not health_df_loaded.empty:
        logger.info(f"Health records loaded successfully. Shape: {health_df_loaded.shape}. Columns: {health_df_loaded.columns.tolist()}")
        if 'date' not in health_df_loaded.columns or not pd.api.types.is_datetime64_any_dtype(health_df_loaded['date']):
            logger.error("CRITICAL: 'date' column in health_df is missing or not datetime after loading!")
        if 'item' not in health_df_loaded.columns:
            logger.warning("WARNING: 'item' column missing in health_df. Supply forecasts will be affected.")
    elif health_df_loaded is not None and health_df_loaded.empty:
        logger.warning("Health records loaded, but the DataFrame is empty.")
    else: # health_df_loaded is None
        logger.error("CRITICAL: load_health_records() returned None.")
        # Create a minimal empty DataFrame to prevent downstream 'NoneType' errors if possible
        health_df_loaded = pd.DataFrame(columns=['date', 'item']) # Add other essential cols

    if iot_df_loaded is not None and not iot_df_loaded.empty:
        logger.info(f"IoT data loaded successfully. Shape: {iot_df_loaded.shape}. Columns: {iot_df_loaded.columns.tolist()}")
    elif iot_df_loaded is not None and iot_df_loaded.empty:
        logger.warning("IoT data loaded, but the DataFrame is empty.")
    else: # iot_df_loaded is None
        logger.error("CRITICAL: load_iot_clinic_environment_data() returned None.")
        iot_df_loaded = pd.DataFrame(columns=['timestamp']) # Minimal empty DF

    return health_df_loaded, iot_df_loaded

health_df, iot_df_clinic = get_clinic_page_data_extended()

# --- DEBUG: Inspect loaded health_df immediately ---
if st.checkbox("Show Raw Health Data Debug (Clinic)", value=False, key="debug_clinic_health_data_raw"):
    st.subheader("Raw `health_df` Inspection (Post-Load)")
    if health_df is not None:
        st.write(f"`health_df` is empty: {health_df.empty}")
        st.write(f"`health_df` columns: `{health_df.columns.tolist()}`")
        if not health_df.empty:
            st.write("`health_df.head()`:")
            st.dataframe(health_df.head())
            if 'date' in health_df.columns:
                st.write(f"`health_df['date']` dtype: {health_df['date'].dtype}, NaNs: {health_df['date'].isna().sum()}")
            else:
                st.error("`health_df` is missing 'date' column!")
            if 'item' in health_df.columns:
                st.write(f"`health_df['item']` dtype: {health_df['item'].dtype}, NaNs: {health_df['item'].isna().sum()}")
            else:
                st.error("`health_df` is missing 'item' column!")
    else:
        st.error("`health_df` is None. Data loading failed critically.")
# --- END DEBUG ---


# --- Main Page ---
# Check if essential data is truly unusable
critical_health_data_missing = health_df is None or \
                               health_df.empty or \
                               'date' not in health_df.columns or \
                               not pd.api.types.is_datetime64_any_dtype(health_df['date'])

critical_iot_data_missing = iot_df_clinic is None or iot_df_clinic.empty

if critical_health_data_missing and critical_iot_data_missing:
    st.error("🚨 CRITICAL Error: Could not load usable health records (missing/empty/'date' column invalid) AND IoT data. Clinic Dashboard cannot be displayed.")
    logger.critical("Dashboard stop: health_df unusable AND iot_df_clinic unusable.")
    st.stop()
elif critical_health_data_missing:
    st.warning("⚠️ WARNING: Health records are missing, empty, or the 'date' column is invalid. Some dashboard features will be unavailable or show no data.")
    logger.warning("Health data is unusable for dashboard. Proceeding with IoT data if available.")
    # Create a minimal empty health_df to prevent errors in functions that expect a DataFrame
    health_df = pd.DataFrame(columns=['date', 'item', 'condition', 'patient_id', 'test_type', 'test_result', 'test_turnaround_days', 'test_date'])


st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility Environment**")
st.markdown("---")

# --- Sidebar Filters & Date Range Setup ---
st.sidebar.header("Clinic Filters")
min_date_cl_page = None
max_date_cl_page = None
default_days_trend = getattr(app_config, 'DEFAULT_DATE_RANGE_DAYS_TREND', 30)
default_start_cl = pd.Timestamp('today').date() - pd.Timedelta(days=default_days_trend - 1)
default_end_cl = pd.Timestamp('today').date()

# Date filter setup requires a non-empty health_df with a valid 'date' column
if health_df is not None and not health_df.empty and \
   'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']) and \
   health_df['date'].notna().any():
    
    min_date_dt_series_cl = health_df['date'].min()
    max_date_dt_series_cl = health_df['date'].max()

    if pd.notna(min_date_dt_series_cl) and pd.notna(max_date_dt_series_cl):
        min_date_cl_page = min_date_dt_series_cl.date()
        max_date_cl_page = max_date_dt_series_cl.date()
        
        if min_date_cl_page > max_date_cl_page: min_date_cl_page = max_date_cl_page
        
        default_end_cl = max_date_cl_page
        default_start_cl_dt = pd.to_datetime(max_date_cl_page) - pd.Timedelta(days=default_days_trend - 1)
        default_start_cl = default_start_cl_dt.date()
        
        if default_start_cl < min_date_cl_page: default_start_cl = min_date_cl_page
        if default_start_cl > default_end_cl: default_start_cl = default_end_cl
    else:
        logger.warning("Min/Max dates in health_df['date'] are NaT. Using system default date range for clinic filter.")
else:
    logger.warning("Health_df is empty, or 'date' column is missing/invalid/all_NaT. Using system default date range for clinic filter.")

if min_date_cl_page is None: min_date_cl_page = pd.Timestamp('today').date() - pd.Timedelta(days=365)
if max_date_cl_page is None: max_date_cl_page = pd.Timestamp('today').date()
if default_start_cl < min_date_cl_page: default_start_cl = min_date_cl_page
if default_end_cl > max_date_cl_page: default_end_cl = max_date_cl_page
if default_start_cl > default_end_cl: default_start_cl = default_end_cl

start_date, end_date = st.sidebar.date_input(
    "Select Date Range for Analysis", 
    value=[default_start_cl, default_end_cl],
    min_value=min_date_cl_page,
    max_value=max_date_cl_page, 
    key="clinic_date_range_final_v4",
    help="Applies to most charts and KPIs unless specified."
)

# --- Filter dataframes based on selected date range ---
# Initialize filtered_df with columns from health_df if available, otherwise a minimal set
health_df_cols = health_df.columns if health_df is not None and not health_df.empty else ['date', 'item', 'condition']
filtered_df = pd.DataFrame(columns=health_df_cols)

if health_df is not None and not health_df.empty and \
   'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
    if start_date and end_date and start_date <= end_date:
        date_mask = (health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)
        filtered_df = health_df[date_mask].copy()
    else: # Date range from widget is invalid or not yet set by user fully
        logger.info("Date range from widget invalid or not fully set for health_df filter. Using all health_df for now.")
        filtered_df = health_df.copy() # Or consider not filtering yet until range is valid
else:
    logger.warning("Cannot filter health_df by date as it's empty or 'date' column is problematic.")
    # filtered_df remains empty with predefined columns

iot_df_cols = iot_df_clinic.columns if iot_df_clinic is not None and not iot_df_clinic.empty else ['timestamp']
filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_cols)

if iot_df_clinic is not None and not iot_df_clinic.empty and \
   'timestamp' in iot_df_clinic.columns and pd.api.types.is_datetime64_any_dtype(iot_df_clinic['timestamp']):
    if start_date and end_date and start_date <= end_date:
        iot_date_mask = (iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)
        filtered_iot_df_clinic = iot_df_clinic[iot_date_mask].copy()
    else:
        logger.info("Date range from widget invalid or not fully set for iot_df_clinic filter. Using all iot_df_clinic for now.")
        filtered_iot_df_clinic = iot_df_clinic.copy()
else:
    logger.warning("Cannot filter iot_df_clinic by date as it's empty or 'timestamp' column is problematic.")

# --- KPIs ---
date_range_str = f"({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')})" if start_date and end_date and start_date <= end_date else "(Full Period or Awaiting Date Filter)"
st.subheader(f"Key Disease Service Metrics {date_range_str}")

# Check if filtered_df is usable for KPIs
if filtered_df is not None and not filtered_df.empty:
    clinic_kpis = get_clinic_summary(filtered_df) # Assumes get_clinic_summary handles potentially missing internal columns
    kpi_cols_disease_cl = st.columns(5)
    # ... (rest of your KPI rendering - ensure .get() is used for robustness) ...
    with kpi_cols_disease_cl[0]: render_kpi_card("TB Sputum Positivity", f"{clinic_kpis.get('tb_sputum_positivity',0.0):.1f}%", "🔬", status="High" if clinic_kpis.get('tb_sputum_positivity',0) > 10 else "Moderate", help_text="Percentage of sputum tests positive for TB among conclusive tests.")
    with kpi_cols_disease_cl[1]: render_kpi_card("Malaria RDT Positivity", f"{clinic_kpis.get('malaria_rdt_positivity',0.0):.1f}%", "🦟", status="High" if clinic_kpis.get('malaria_rdt_positivity',0) > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Moderate", help_text=f"Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%. RDT positivity for Malaria.")
    with kpi_cols_disease_cl[2]: render_kpi_card("STI Tests Pending", str(clinic_kpis.get('sti_tests_pending',0)), "🧪", status="Moderate" if clinic_kpis.get('sti_tests_pending',0) > 5 else "Low", help_text="Number of key STI tests currently pending results.")
    with kpi_cols_disease_cl[3]: 
        hiv_icon = "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='28' alt='HIV' style='vertical-align: middle;'>"
        render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), hiv_icon, icon_is_html=True, help_text="Number of conclusive HIV tests conducted in the period.")
    with kpi_cols_disease_cl[4]: render_kpi_card("Key Drug Stockouts", str(clinic_kpis.get('critical_disease_supply_items',0)), "💊", status="High" if clinic_kpis.get('critical_disease_supply_items',0) > 0 else "Low", help_text=f"Key disease drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days stock.")

else:
    st.info("No health record data available for the selected period (or base data unusable) to display disease service KPIs.")

st.subheader(f"Clinic Environment Snapshot {date_range_str}")
if filtered_iot_df_clinic is not None and not filtered_iot_df_clinic.empty:
    env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic)
    kpi_cols_env_cl = st.columns(4)
    # ... (rest of your IoT KPI rendering) ...
    with kpi_cols_env_cl[0]: render_kpi_card("Avg. CO2", f"{env_summary_clinic.get('avg_co2',0):.0f} ppm", "💨", status="High" if env_summary_clinic.get('co2_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {env_summary_clinic.get('co2_alert_rooms',0)}")
    with kpi_cols_env_cl[1]: render_kpi_card("Avg. PM2.5", f"{env_summary_clinic.get('avg_pm25',0):.1f} µg/m³", "🌫️", status="High" if env_summary_clinic.get('pm25_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {env_summary_clinic.get('pm25_alert_rooms',0)}")
    with kpi_cols_env_cl[2]: render_kpi_card("Avg. Occupancy", f"{env_summary_clinic.get('avg_occupancy',0):.1f}", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
    with kpi_cols_env_cl[3]: render_kpi_card("Sanitizer Use", f"{env_summary_clinic.get('avg_sanitizer_use_hr',0):.1f}/hr", "🧴", status="Low" if env_summary_clinic.get('avg_sanitizer_use_hr',0) < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High", help_text="Avg. dispenses/hr/unit.")

else:
    st.info("No IoT environmental data available for the selected period (or base data unusable) to display environmental KPIs.")
st.markdown("---")

# --- Tabs for Detailed Views ---
tab_tests_cl, tab_supplies_cl, tab_patients_cl, tab_environment_cl = st.tabs([
    "🔬 Disease Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Clinic Environment"
])

with tab_tests_cl:
    st.subheader("Disease-Specific Test Results & Performance")
    required_cols_tests = ['test_type', 'test_result', 'patient_id']
    if filtered_df is not None and not filtered_df.empty and all(col in filtered_df.columns for col in required_cols_tests):
        # ... (rest of your tab_tests_cl logic - it was mostly fine, ensure .get() for robustness) ...
            col_test1_cl, col_test2_cl = st.columns([0.4, 0.6]) 
            with col_test1_cl:
                tb_tests_df = filtered_df[filtered_df['test_type'].astype(str).str.contains("Sputum|GeneXpert", case=False, na=False)].copy()
                if not tb_tests_df.empty:
                    tb_results_summary = tb_tests_df.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
                    tb_results_summary.columns = ['Test Result', 'Count']
                    tb_results_summary = tb_results_summary[~tb_results_summary['Test Result'].astype(str).isin(['Unknown', 'N/A', 'nan', 'Pending'])]
                    if not tb_results_summary.empty: 
                        st.plotly_chart(plot_donut_chart(tb_results_summary, 'Test Result', 'Count', "TB Test Result Distribution"), use_container_width=True)
                    else: st.caption("No conclusive TB test result data for pie chart.")
                else: st.caption("No TB tests found in selected period.")
            
            with col_test2_cl:
                if 'test_turnaround_days' in filtered_df.columns and 'test_date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['test_date']):
                    turnaround_trend_df_cl = filtered_df.dropna(subset=['test_turnaround_days', 'test_date']).copy()
                    # test_date should already be datetime if it passed the main health_df check
                    if not turnaround_trend_df_cl.empty:
                        turnaround_trend_cl = get_trend_data(turnaround_trend_df_cl,'test_turnaround_days', period='D', date_col='test_date')
                        if not turnaround_trend_cl.empty: 
                            st.plotly_chart(plot_annotated_line_chart(
                                turnaround_trend_cl, "Daily Avg. Test Turnaround Time", 
                                y_axis_title="Days", target_line=app_config.TARGET_TEST_TURNAROUND_DAYS,
                                height=app_config.DEFAULT_PLOT_HEIGHT
                            ), use_container_width=True)
                        else: st.caption("No aggregated turnaround time data for trend.")
                    else: st.caption("No valid raw turnaround time data for trend (after date processing).")
                else: 
                    st.caption("Required columns ('test_turnaround_days', 'test_date' as datetime) missing for turnaround trend.")
    else: 
        st.info(f"Health records are empty, or missing one or more required columns for Disease Testing tab: {', '.join(required_cols_tests)}.")


with tab_supplies_cl:
    st.subheader("Supply Levels & Forecast (Key Disease Drugs)")
    # Supply forecast uses the original, unfiltered health_df for a complete historical view
    # THE CRITICAL CHECK:
    if health_df is not None and not health_df.empty and \
       'item' in health_df.columns and \
       'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        
        supply_forecast_df_all_items = get_supply_forecast_data(health_df) # Assumes health_df is now good
        
        if supply_forecast_df_all_items is not None and not supply_forecast_df_all_items.empty:
            key_drug_substrings = getattr(app_config, 'KEY_DRUG_SUBSTRINGS_SUPPLY', ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin'])
            available_key_items_cl = sorted([
                item for item in supply_forecast_df_all_items['item'].unique() 
                if pd.notna(item) and any(sub.lower() in str(item).lower() for sub in key_drug_substrings)
            ])
            if not available_key_items_cl: 
                st.info("No forecast data available for key disease drugs based on current stock definitions or item names.")
            else:
                selected_item_clinic_tab = st.selectbox("Select Key Drug for Forecast:", available_key_items_cl, key="supply_item_select_clinic_final_v3") # Incremented key
                if selected_item_clinic_tab: 
                    item_forecast_df_cl = supply_forecast_df_all_items[supply_forecast_df_all_items['item'] == selected_item_clinic_tab].copy()
                    if not item_forecast_df_cl.empty:
                        if 'date' in item_forecast_df_cl.columns: # Ensure 'date' is suitable for index
                            item_forecast_df_cl.sort_values('date', inplace=True)
                            item_forecast_df_cl.set_index('date', inplace=True)
                        
                        st.plotly_chart(plot_annotated_line_chart(
                            item_forecast_df_cl['forecast_days'],f"Forecast: {selected_item_clinic_tab} (Days of Supply Remaining)",
                            y_axis_title="Days of Supply",target_line=app_config.CRITICAL_SUPPLY_DAYS, 
                            target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)",show_ci=True, 
                            lower_bound_series=item_forecast_df_cl.get('lower_ci'), 
                            upper_bound_series=item_forecast_df_cl.get('upper_ci'),
                            height=getattr(app_config, 'DEFAULT_PLOT_HEIGHT', 380) + 50
                        ), use_container_width=True)
                    else: st.info(f"No forecast data found for {selected_item_clinic_tab}.")
        else: 
            st.warning("Output of `get_supply_forecast_data` is empty or None. Cannot display supply forecasts.")
    else: 
        # This is where your original error message comes from
        st.error("CRITICAL FOR SUPPLY TAB: Cannot generate supply forecasts. The base health data (`health_df`) is empty, or missing a valid 'item' column, or missing a valid 'date' (datetime) column.")
        logger.error("Supply tab cannot proceed: health_df missing 'item' or valid 'date' column.")


with tab_patients_cl:
    st.subheader("Patient Load by Key Conditions")
    required_cols_patients = ['condition', 'date', 'patient_id']
    if filtered_df is not None and not filtered_df.empty and \
       all(col in filtered_df.columns for col in required_cols_patients) and \
       pd.api.types.is_datetime64_any_dtype(filtered_df['date']): # Ensure date is still datetime
        
        key_conditions_list = getattr(app_config, 'KEY_CONDITIONS_FOR_TRENDS', ['TB', 'Malaria', 'HIV-Positive'])
        key_conditions_clinic_df = filtered_df[filtered_df['condition'].astype(str).isin(key_conditions_list)].copy()
        
        if not key_conditions_clinic_df.empty:
            patient_condition_summary = key_conditions_clinic_df.groupby([pd.Grouper(key='date', freq='D'), 'condition'])['patient_id'].nunique().reset_index()
            patient_condition_summary.rename(columns={'patient_id': 'patient_count'}, inplace=True)
            if not patient_condition_summary.empty:
                patient_pivot = patient_condition_summary.pivot_table(index='date', columns='condition', values='patient_count', fill_value=0).reset_index()
                patient_melt = patient_pivot.melt(id_vars='date', var_name='condition', value_name='patient_count')
                st.plotly_chart(plot_bar_chart(
                    patient_melt, x_col='date', y_col='patient_count', 
                    title="Daily Patient Count by Key Condition", color_col='condition', 
                    barmode='stack', height=getattr(app_config, 'DEFAULT_PLOT_HEIGHT', 380)+50
                ), use_container_width=True)
            else: st.caption("No patient data for key conditions in selected period to display chart.")
        else: st.caption("No patients with key conditions found in the selected period.")
    else: st.info(f"Health records (filtered) are empty or missing required columns for Patient Load tab: {', '.join(required_cols_patients)} or 'date' is not datetime.")
    
    st.markdown("---"); st.markdown("###### Flagged Patient Cases for Review (Focused Diseases)")
    if filtered_df is not None and not filtered_df.empty:
        # get_patient_alerts_for_clinic should handle missing internal columns gracefully
        flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS.get('moderate', 60))
        if flagged_patients_clinic_df is not None and not flagged_patients_clinic_df.empty:
            key_conditions_list_alerts = getattr(app_config, 'KEY_CONDITIONS_FOR_TRENDS', ['TB', 'Malaria', 'HIV-Positive'])
            # Use .get() for 'condition' as its presence depends on get_patient_alerts_for_clinic
            flagged_focused_df = flagged_patients_clinic_df[flagged_patients_clinic_df.get('condition', pd.Series(dtype=str)).isin(key_conditions_list_alerts)]
            if not flagged_focused_df.empty:
                display_cols = ['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result']
                actual_display_cols = [col for col in display_cols if col in flagged_focused_df.columns]
                # Ensure 'ai_risk_score' is present before trying to sort or format it
                sort_col = 'ai_risk_score' if 'ai_risk_score' in actual_display_cols else (actual_display_cols[0] if actual_display_cols else None)
                column_config_alerts = { "ai_risk_score": st.column_config.NumberColumn(format="%.0f")} if 'ai_risk_score' in actual_display_cols else {}

                if sort_col:
                    st.dataframe(flagged_focused_df[actual_display_cols].sort_values(by=sort_col, ascending=False).head(10), 
                                use_container_width=True, column_config=column_config_alerts)
                else:
                     st.dataframe(flagged_focused_df[actual_display_cols].head(10), use_container_width=True, column_config=column_config_alerts)
            else: st.info("No flagged patients matching key diseases found in the selected period.")
        else: st.info("No specific patient cases flagged for review in the selected period.")
    else: st.info("Filtered health records are empty; cannot generate patient alerts.")


with tab_environment_cl:
    st.subheader("Clinic Environmental Monitoring")
    required_cols_env_trends = ['timestamp'] # avg_co2_ppm, waiting_room_occupancy are checked inside
    if filtered_iot_df_clinic is not None and not filtered_iot_df_clinic.empty and \
       all(col in filtered_iot_df_clinic.columns for col in required_cols_env_trends) and \
       pd.api.types.is_datetime64_any_dtype(filtered_iot_df_clinic['timestamp']):

        # Use env_summary_clinic calculated for KPIs if available and based on the same filtered_iot_df_clinic
        # Or recalculate if necessary:
        current_env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic)

        st.markdown(f"**Alerts Summary (Latest readings in period):** {current_env_summary_clinic.get('co2_alert_rooms',0)} room(s) with high CO2; {current_env_summary_clinic.get('pm25_alert_rooms',0)} room(s) with high PM2.5.")
        if current_env_summary_clinic.get('high_occupancy_alert'): st.warning(f"High waiting room occupancy detected (above {app_config.TARGET_WAITING_ROOM_OCCUPANCY}). Consider patient flow adjustments.")
        
        env_trend_cols_cl_tab = st.columns(2)
        with env_trend_cols_cl_tab[0]:
            if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
                co2_trend_cl_tab = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not co2_trend_cl_tab.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend_cl_tab, "Hourly Avg. CO2 Levels", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=getattr(app_config, 'DEFAULT_PLOT_HEIGHT', 380)-20), use_container_width=True)
                else: st.caption("No CO2 trend data.")
            else: st.caption("Avg. CO2 data ('avg_co2_ppm') missing for trend.")
        with env_trend_cols_cl_tab[1]:
            if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
                occupancy_trend_cl_tab = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not occupancy_trend_cl_tab.empty: st.plotly_chart(plot_annotated_line_chart(occupancy_trend_cl_tab, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, height=getattr(app_config, 'DEFAULT_PLOT_HEIGHT', 380)-20), use_container_width=True)
                else: st.caption("No occupancy trend.")
            else: st.caption("Occupancy data ('waiting_room_occupancy') missing for trend.")
        
        st.subheader("Latest Room Readings (in selected period)")
        latest_room_display_cols = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'waiting_room_occupancy']
        actual_latest_cols = [col for col in latest_room_display_cols if col in filtered_iot_df_clinic.columns]
        required_latest_cols_check = ['timestamp', 'clinic_id', 'room_name']

        if all(col in actual_latest_cols for col in required_latest_cols_check):
            latest_room_readings_cl = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
            if not latest_room_readings_cl.empty: 
                st.dataframe(latest_room_readings_cl[actual_latest_cols].tail(10), use_container_width=True, height=280) 
            else: st.caption("No detailed room readings for selected period after filtering.")
        else: st.caption(f"Essential columns missing for detailed room readings. Need: {', '.join(required_latest_cols_check)}. Available: {', '.join(actual_latest_cols)}")
    else: st.info("No clinic environmental data (filtered) available for this tab, or 'timestamp' column is problematic.")
