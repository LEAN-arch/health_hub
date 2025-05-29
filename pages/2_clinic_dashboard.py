# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data, # Corrected from load_iot_clinic_environment_dataYou
    get_clinic_summary, get_clinic_environmental_summary,
    get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Get logger for this page

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) # Corrected from prompt artifact
    else:
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600)
def get_clinic_page_data_extended():
    logger.info("Loading extended data for Clinic Dashboard...")
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    logger.info(f"Health records loaded: {not health_df.empty}, IoT data loaded: {not iot_df.empty}")
    return health_df, iot_df

health_df, iot_df_clinic = get_clinic_page_data_extended()

# --- Main Page ---
if health_df.empty and (iot_df_clinic is None or iot_df_clinic.empty): # Enhanced check for iot_df_clinic
    st.error("🚨 CRITICAL Error: Could not load health records or IoT data. Clinic Dashboard cannot be displayed.")
    st.stop() # Stop if no data at all
else:
    st.title("🏥 Clinic Operations & Environmental Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility Environment**")
    st.markdown("---")

    # --- Sidebar Filters & Date Range Setup ---
    st.sidebar.header("Clinic Filters")
    min_date_cl_page = None
    max_date_cl_page = None
    # Initialize defaults, to be refined if data is available
    default_start_cl = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
    default_end_cl = pd.Timestamp('today').date()

    if not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        if health_df['date'].notna().any():
            min_date_dt_series_cl = health_df['date'].min()
            max_date_dt_series_cl = health_df['date'].max()

            if pd.notna(min_date_dt_series_cl) and pd.notna(max_date_dt_series_cl):
                min_date_cl_page = min_date_dt_series_cl.date()
                max_date_cl_page = max_date_dt_series_cl.date()
                
                if min_date_cl_page > max_date_cl_page: # Should not happen with valid min/max calls
                    min_date_cl_page = max_date_cl_page # pragma: no cover
                
                # Recalculate defaults based on actual data range
                default_end_cl = max_date_cl_page
                default_start_cl_dt = pd.to_datetime(max_date_cl_page) - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
                default_start_cl = default_start_cl_dt.date()
                
                if default_start_cl < min_date_cl_page:
                    default_start_cl = min_date_cl_page # Corrected typo: default_start -> default_start_cl
                if default_start_cl > default_end_cl: # Safety if data range is very short
                    default_start_cl = default_end_cl # pragma: no cover
            else:
                logger.warning("All 'date' values in health_df are NaT. Using system default date range for clinic filter.")
                # default_start_cl, default_end_cl remain system defaults
        else:
            logger.warning("Health_df 'date' column has no valid dates (all NaT). Using system default date range for clinic filter.")
            # default_start_cl, default_end_cl remain system defaults
    else:
        logger.warning("Health_df empty or 'date' column missing/invalid. Using system default date range for clinic filter.")
        # default_start_cl, default_end_cl remain system defaults
    
    # Final fallback for min/max_value of date_input if health_df processing yielded no dates
    if min_date_cl_page is None:
        min_date_cl_page = pd.Timestamp('today').date() - pd.Timedelta(days=365) # Wider fallback for picker
    if max_date_cl_page is None:
        max_date_cl_page = pd.Timestamp('today').date()
    
    # Ensure default_start_cl is not before min_date_cl_page (if min_date_cl_page was set from fallback)
    if default_start_cl < min_date_cl_page : default_start_cl = min_date_cl_page
    # Ensure default_end_cl is not after max_date_cl_page (if max_date_cl_page was set from fallback)
    if default_end_cl > max_date_cl_page : default_end_cl = max_date_cl_page
    # Ensure start is not after end for default value list
    if default_start_cl > default_end_cl: default_start_cl = default_end_cl


    start_date, end_date = st.sidebar.date_input(
        "Select Date Range for Analysis", 
        value=[default_start_cl, default_end_cl],
        min_value=min_date_cl_page,
        max_value=max_date_cl_page, 
        key="clinic_date_range_final_v3", # Unique key
        help="Applies to most charts and KPIs unless specified."
    )
    
    # Filter dataframes based on selected date range
    filtered_df = pd.DataFrame(columns=health_df.columns if not health_df.empty else None) # Initialize with columns
    if start_date and end_date and start_date <= end_date:
        if not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
            date_mask = (health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)
            filtered_df = health_df[date_mask].copy()
        elif not health_df.empty: # Date column issue but data exists
             logger.warning("Health data 'date' column issue after filter setup. Using full health data for selected range if possible, or all data.")
             filtered_df = health_df.copy() # Fallback to all data if specific filtering fails post-selection
    elif not health_df.empty: # Date range from widget is invalid (e.g. start > end, though widget prevents this)
        logger.warning("Date range from widget invalid for health_df filter, using all health_df.")
        filtered_df = health_df.copy()

    filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic.columns if iot_df_clinic is not None and not iot_df_clinic.empty else None)
    if start_date and end_date and start_date <= end_date:
        if iot_df_clinic is not None and not iot_df_clinic.empty and \
           'timestamp' in iot_df_clinic.columns and \
           pd.api.types.is_datetime64_any_dtype(iot_df_clinic['timestamp']):
            iot_date_mask = (iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)
            filtered_iot_df_clinic = iot_df_clinic[iot_date_mask].copy()
        elif iot_df_clinic is not None and not iot_df_clinic.empty: # Timestamp column issue or data exists
            logger.warning("IoT data 'timestamp' column issue or not datetime. Using full IoT data for selected range if possible, or all data.")
            filtered_iot_df_clinic = iot_df_clinic.copy() # Fallback
    elif iot_df_clinic is not None and not iot_df_clinic.empty: # Date range from widget is invalid
        logger.warning("Date range from widget invalid for iot_df_clinic filter, using all iot_df_clinic.")
        filtered_iot_df_clinic = iot_df_clinic.copy()

    # --- KPIs ---
    date_range_str = f"({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')})" if start_date and end_date else "(Full Period)"
    st.subheader(f"Key Disease Service Metrics {date_range_str}")
    if not filtered_df.empty:
        clinic_kpis = get_clinic_summary(filtered_df)
        kpi_cols_disease_cl = st.columns(5)
        with kpi_cols_disease_cl[0]: render_kpi_card("TB Sputum Positivity", f"{clinic_kpis.get('tb_sputum_positivity',0.0):.1f}%", "🔬", status="High" if clinic_kpis.get('tb_sputum_positivity',0) > 10 else "Moderate", help_text="Percentage of sputum tests positive for TB among conclusive tests.")
        with kpi_cols_disease_cl[1]: render_kpi_card("Malaria RDT Positivity", f"{clinic_kpis.get('malaria_rdt_positivity',0.0):.1f}%", "🦟", status="High" if clinic_kpis.get('malaria_rdt_positivity',0) > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Moderate", help_text=f"Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%. RDT positivity for Malaria.")
        with kpi_cols_disease_cl[2]: render_kpi_card("STI Tests Pending", str(clinic_kpis.get('sti_tests_pending',0)), "🧪", status="Moderate" if clinic_kpis.get('sti_tests_pending',0) > 5 else "Low", help_text="Number of key STI tests currently pending results.")
        with kpi_cols_disease_cl[3]: 
            hiv_icon = "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='28' alt='HIV' style='vertical-align: middle;'>"
            render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), hiv_icon, icon_is_html=True, help_text="Number of conclusive HIV tests conducted in the period.")
        with kpi_cols_disease_cl[4]: render_kpi_card("Key Drug Stockouts", str(clinic_kpis.get('critical_disease_supply_items',0)), "💊", status="High" if clinic_kpis.get('critical_disease_supply_items',0) > 0 else "Low", help_text=f"Key disease drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days stock.")
    else:
        st.info("No health record data available for the selected period to display disease service KPIs.")

    st.subheader(f"Clinic Environment Snapshot {date_range_str}")
    if not filtered_iot_df_clinic.empty:
        env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic)
        kpi_cols_env_cl = st.columns(4)
        with kpi_cols_env_cl[0]: render_kpi_card("Avg. CO2", f"{env_summary_clinic.get('avg_co2',0):.0f} ppm", "💨", status="High" if env_summary_clinic.get('co2_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {env_summary_clinic.get('co2_alert_rooms',0)}")
        with kpi_cols_env_cl[1]: render_kpi_card("Avg. PM2.5", f"{env_summary_clinic.get('avg_pm25',0):.1f} µg/m³", "🌫️", status="High" if env_summary_clinic.get('pm25_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {env_summary_clinic.get('pm25_alert_rooms',0)}")
        with kpi_cols_env_cl[2]: render_kpi_card("Avg. Occupancy", f"{env_summary_clinic.get('avg_occupancy',0):.1f}", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
        with kpi_cols_env_cl[3]: render_kpi_card("Sanitizer Use", f"{env_summary_clinic.get('avg_sanitizer_use_hr',0):.1f}/hr", "🧴", status="Low" if env_summary_clinic.get('avg_sanitizer_use_hr',0) < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High", help_text="Avg. dispenses/hr/unit.")
    else:
        st.info("No IoT environmental data available for the selected period to display environmental KPIs.")
    st.markdown("---")

    # --- Tabs for Detailed Views ---
    tab_tests_cl, tab_supplies_cl, tab_patients_cl, tab_environment_cl = st.tabs([
        "🔬 Disease Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Clinic Environment"
    ])

    with tab_tests_cl:
        st.subheader("Disease-Specific Test Results & Performance")
        if not filtered_df.empty and 'test_type' in filtered_df.columns and \
           'test_result' in filtered_df.columns and 'patient_id' in filtered_df.columns:
            col_test1_cl, col_test2_cl = st.columns([0.4, 0.6]) 
            with col_test1_cl:
                # Using filtered_df['test_type'] assuming the outer 'if' guarantees its existence
                tb_tests_df = filtered_df[filtered_df['test_type'].astype(str).str.contains("Sputum|GeneXpert", case=False, na=False)].copy()
                if not tb_tests_df.empty:
                    tb_results_summary = tb_tests_df.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
                    tb_results_summary.columns = ['Test Result', 'Count']
                    tb_results_summary = tb_results_summary[~tb_results_summary['Test Result'].astype(str).isin(['Unknown', 'N/A', 'nan', 'Pending'])] # Ensure comparison with strings
                    if not tb_results_summary.empty: 
                        st.plotly_chart(plot_donut_chart(tb_results_summary, 'Test Result', 'Count', "TB Test Result Distribution"), use_container_width=True)
                    else: st.caption("No conclusive TB test result data for pie chart.")
                else: st.caption("No TB tests found in selected period.")
            
            with col_test2_cl:
                if 'test_turnaround_days' in filtered_df.columns and 'test_date' in filtered_df.columns:
                    turnaround_trend_df_cl = filtered_df.dropna(subset=['test_turnaround_days', 'test_date']).copy()
                    if not turnaround_trend_df_cl.empty and pd.api.types.is_datetime64_any_dtype(turnaround_trend_df_cl['test_date']): # test_date should be datetime
                        # Ensure 'test_date' is datetime after potential subsetting/copying, though it should be from health_df
                        turnaround_trend_df_cl['test_date'] = pd.to_datetime(turnaround_trend_df_cl['test_date'], errors='coerce')
                        turnaround_trend_df_cl.dropna(subset=['test_date'], inplace=True) # Drop if coerce made NaT

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
                    else: st.caption("No raw turnaround time data (or 'test_date' not datetime) for trend.")
                else: 
                    st.caption("Required columns ('test_turnaround_days', 'test_date') missing for turnaround trend.")
        else: 
            st.info("Health records are empty or missing key columns ('test_type', 'test_result', 'patient_id') for this tab.")


    with tab_supplies_cl:
        st.subheader("Supply Levels & Forecast (Key Disease Drugs)")
        # Supply forecast generally uses all historical data (health_df, not filtered_df)
        if not health_df.empty and 'item' in health_df.columns and 'date' in health_df.columns:
            supply_forecast_df_all_items = get_supply_forecast_data(health_df) 
            if not supply_forecast_df_all_items.empty:
                # Consider moving key_drug_substrings to app_config
                key_drug_substrings = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Benznidazole', 'ORS', 'Oxygen', 'Metronidazole', 'Gardasil'] 
                available_key_items_cl = sorted([
                    item for item in supply_forecast_df_all_items['item'].unique() 
                    if pd.notna(item) and any(sub.lower() in str(item).lower() for sub in key_drug_substrings)
                ])
                if not available_key_items_cl: 
                    st.info("No forecast data available for key disease drugs based on current stock definitions.")
                else:
                    selected_item_clinic_tab = st.selectbox("Select Key Drug for Forecast:", available_key_items_cl, key="supply_item_select_clinic_final_v2")
                    if selected_item_clinic_tab: 
                        item_forecast_df_cl = supply_forecast_df_all_items[supply_forecast_df_all_items['item'] == selected_item_clinic_tab].copy()
                        if not item_forecast_df_cl.empty:
                            # Ensure 'date' is index for plotting if get_supply_forecast_data doesn't already do it
                            if 'date' in item_forecast_df_cl.columns:
                                item_forecast_df_cl.sort_values('date', inplace=True)
                                item_forecast_df_cl.set_index('date', inplace=True)
                            
                            st.plotly_chart(plot_annotated_line_chart(
                                item_forecast_df_cl['forecast_days'],f"Forecast: {selected_item_clinic_tab} (Days of Supply Remaining)",
                                y_axis_title="Days of Supply",target_line=app_config.CRITICAL_SUPPLY_DAYS, 
                                target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)",show_ci=True, 
                                lower_bound_series=item_forecast_df_cl.get('lower_ci'), # Use .get for optional CI
                                upper_bound_series=item_forecast_df_cl.get('upper_ci'), # Use .get for optional CI
                                height=app_config.DEFAULT_PLOT_HEIGHT + 50
                            ), use_container_width=True)
                        else: st.info(f"No forecast data found for {selected_item_clinic_tab}.") # Should not happen if in available_key_items_cl
            else: st.caption("No overall supply data available to generate forecasts (output of get_supply_forecast_data is empty).")
        else: st.info("Cannot generate supply forecasts: Base health data is empty or missing 'item'/'date' columns.")


    with tab_patients_cl:
        st.subheader("Patient Load by Key Conditions")
        if not filtered_df.empty and 'condition' in filtered_df.columns and \
           'date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['date']) and \
           'patient_id' in filtered_df.columns:
            key_conditions_clinic_df = filtered_df[filtered_df['condition'].astype(str).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)].copy()
            if not key_conditions_clinic_df.empty:
                # Grouper requires 'date' column to be datetime and present
                patient_condition_summary = key_conditions_clinic_df.groupby([pd.Grouper(key='date', freq='D'), 'condition'])['patient_id'].nunique().reset_index()
                patient_condition_summary.rename(columns={'patient_id': 'patient_count'}, inplace=True)
                if not patient_condition_summary.empty:
                    patient_pivot = patient_condition_summary.pivot_table(index='date', columns='condition', values='patient_count', fill_value=0).reset_index()
                    patient_melt = patient_pivot.melt(id_vars='date', var_name='condition', value_name='patient_count')
                    st.plotly_chart(plot_bar_chart(
                        patient_melt, x_col='date', y_col='patient_count', 
                        title="Daily Patient Count by Key Condition", color_col='condition', 
                        barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT+50
                    ), use_container_width=True)
                else: st.caption("No patient data for key conditions in selected period to display chart.")
            else: st.caption("No patients with key conditions found in the selected period.")
        else: st.info("Health records are empty or missing required columns ('condition', 'date', 'patient_id') for patient load analysis.")
        
        st.markdown("---"); st.markdown("###### Flagged Patient Cases for Review (Focused Diseases)")
        if not filtered_df.empty:
            flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS['moderate'])
            if not flagged_patients_clinic_df.empty:
                # Ensure 'condition' column exists before trying to filter on it. get() is safer.
                flagged_focused_df = flagged_patients_clinic_df[flagged_patients_clinic_df.get('condition', pd.Series(dtype=str)).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)]
                if not flagged_focused_df.empty:
                    display_cols = ['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result']
                    # Ensure all display_cols actually exist in flagged_focused_df
                    actual_display_cols = [col for col in display_cols if col in flagged_focused_df.columns]
                    st.dataframe(flagged_focused_df[actual_display_cols].sort_values(by='ai_risk_score', ascending=False).head(10), 
                                 use_container_width=True, column_config={ "ai_risk_score": st.column_config.NumberColumn(format="%.0f")}) # Format as integer
                else: st.info("No flagged patients with key diseases found in the selected period.")
            else: st.info("No specific patient cases flagged for review in the selected period (output of get_patient_alerts_for_clinic is empty).")
        else: st.info("Health records are empty, cannot generate patient alerts.")


    with tab_environment_cl:
        st.subheader("Clinic Environmental Monitoring")
        if not filtered_iot_df_clinic.empty:
            # env_summary_clinic (used for KPIs) is calculated based on filtered_iot_df_clinic
            # Re-calculate here if needed, or ensure it's passed/accessible if calculated once at top
            # For simplicity, let's assume env_summary_clinic calculated for KPIs is sufficient for this text
            # If env_summary_clinic was defined in a broader scope based on filtered_iot_df_clinic:
            if 'env_summary_clinic' in locals() and env_summary_clinic: # Check if it was calculated and not empty
                 st.markdown(f"**Alerts Summary (Latest readings in period):** {env_summary_clinic.get('co2_alert_rooms',0)} room(s) with high CO2; {env_summary_clinic.get('pm25_alert_rooms',0)} room(s) with high PM2.5.")
                 if env_summary_clinic.get('high_occupancy_alert'): st.warning(f"High waiting room occupancy detected (above {app_config.TARGET_WAITING_ROOM_OCCUPANCY}). Consider patient flow adjustments.")
            
            env_trend_cols_cl_tab = st.columns(2)
            with env_trend_cols_cl_tab[0]:
                if 'avg_co2_ppm' in filtered_iot_df_clinic.columns and 'timestamp' in filtered_iot_df_clinic.columns:
                    co2_trend_cl_tab = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                    if not co2_trend_cl_tab.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend_cl_tab, "Hourly Avg. CO2 Levels", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
                    else: st.caption("No CO2 trend data.")
                else: st.caption("CO2 or timestamp data missing for trend.")
            with env_trend_cols_cl_tab[1]:
                if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns and 'timestamp' in filtered_iot_df_clinic.columns:
                    occupancy_trend_cl_tab = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                    if not occupancy_trend_cl_tab.empty: st.plotly_chart(plot_annotated_line_chart(occupancy_trend_cl_tab, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
                    else: st.caption("No occupancy trend.")
                else: st.caption("Occupancy or timestamp data missing for trend.")
            
            st.subheader("Latest Room Readings (in selected period)")
            latest_room_cols = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'waiting_room_occupancy']
            actual_latest_cols = [col for col in latest_room_cols if col in filtered_iot_df_clinic.columns]

            if 'timestamp' in actual_latest_cols and 'clinic_id' in actual_latest_cols and 'room_name' in actual_latest_cols :
                latest_room_readings_cl = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
                if not latest_room_readings_cl.empty: 
                    st.dataframe(latest_room_readings_cl[actual_latest_cols].tail(10), use_container_width=True, height=280) 
                else: st.caption("No detailed room readings for selected period.")
            else: st.caption("Essential columns (clinic_id, room_name, timestamp) missing for detailed room readings.")
        else: st.info("No clinic environmental data available for the selected period.")
