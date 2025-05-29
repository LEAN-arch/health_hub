# health_hub/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
import numpy as np # Added for potential np.nan usage
from config import app_config # Centralized configuration
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data, # Robust data loaders
    get_clinic_summary, get_clinic_environmental_summary,    # KPI calculators
    get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic # Analytics functions
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart # Plotting helpers
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Logger for this page

@st.cache_resource # Cache CSS loading
def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("Clinic Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"Clinic Dashboard: CSS file not found at {app_config.STYLE_CSS}. Default Streamlit styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS) # Cache data loading
def get_clinic_dashboard_data():
    logger.info("Clinic Dashboard: Attempting to load health records and IoT data...")
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()

    if health_df.empty: logger.warning("Clinic Dashboard: Health records are empty or failed to load.")
    else: logger.info(f"Clinic Dashboard: Successfully loaded {len(health_df)} health records.")
    if iot_df.empty: logger.warning("Clinic Dashboard: IoT data is empty or failed to load.")
    else: logger.info(f"Clinic Dashboard: Successfully loaded {len(iot_df)} IoT records.")
    return health_df, iot_df

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data()

# --- Main Page Rendering ---
critical_data_missing_flag = False 
if health_df_clinic_main is None or health_df_clinic_main.empty:
    st.warning("⚠️ **Health records data is currently unavailable.** Some dashboard features related to patient services, testing, and supply chain will be limited or show no data.")
    logger.warning("Clinic Dashboard cannot display full health-related metrics: health_df_clinic_main is None or empty.")
    health_df_clinic_main = pd.DataFrame(columns=['date', 'item', 'condition', 'patient_id', 'test_type', 'test_result', 'test_turnaround_days', 'stock_on_hand', 'consumption_rate_per_day', 'ai_risk_score'])
    critical_data_missing_flag = True

if iot_df_clinic_main is None or iot_df_clinic_main.empty:
    st.info("ℹ️ IoT environmental data is unavailable. Clinic environment monitoring section will not be displayed.")
    logger.info("Clinic Dashboard: iot_df_clinic_main is None or empty. Environmental metrics will be skipped.")
    iot_df_clinic_main = pd.DataFrame(columns=['timestamp', 'avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'sanitizer_dispenses_per_hour', 'avg_noise_db', 'clinic_id', 'room_name', 'zone_id'])

if critical_data_missing_flag and iot_df_clinic_main.empty:
     st.error("🚨 **CRITICAL Error:** Both Health records and essential IoT data are unavailable. Clinic Dashboard cannot be displayed.")
     logger.critical("Clinic Dashboard cannot render: both health_df and iot_df are unusable/empty.")
     st.stop()


st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Monitoring Service Efficiency, Quality of Care, Resource Management, and Facility Environment**")
st.markdown("---")

# --- Sidebar Filters & Date Range Setup ---
st.sidebar.header("🗓️ Clinic Filters")

all_valid_timestamps = []

if 'date' in health_df_clinic_main.columns and not health_df_clinic_main.empty:
    if pd.api.types.is_list_like(health_df_clinic_main['date']): # Check if it's list-like before making a Series
        s_health_dates = pd.to_datetime(pd.Series(health_df_clinic_main['date']), errors='coerce')
        valid_health_timestamps = s_health_dates.dropna()
        if not valid_health_timestamps.empty:
            all_valid_timestamps.extend(valid_health_timestamps.tolist())

if 'timestamp' in iot_df_clinic_main.columns and not iot_df_clinic_main.empty:
    if pd.api.types.is_list_like(iot_df_clinic_main['timestamp']):
        s_iot_timestamps = pd.to_datetime(pd.Series(iot_df_clinic_main['timestamp']), errors='coerce')
        valid_iot_timestamps = s_iot_timestamps.dropna()
        if not valid_iot_timestamps.empty:
            all_valid_timestamps.extend(valid_iot_timestamps.tolist())

if not all_valid_timestamps:
    min_date_data_clinic = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 3)
    max_date_data_clinic = pd.Timestamp('today').date()
    logger.warning("Clinic Dashboard: No valid dates found in any dataset for filter range. Using wide fallback.")
else:
    min_date_ts = min(all_valid_timestamps)
    max_date_ts = max(all_valid_timestamps)
    min_date_data_clinic = min_date_ts.date()
    max_date_data_clinic = max_date_ts.date()

default_start_date_clinic = max_date_data_clinic - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_date_clinic < min_date_data_clinic : default_start_date_clinic = min_date_data_clinic
if default_start_date_clinic > max_date_data_clinic : default_start_date_clinic = max_date_data_clinic

selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=[default_start_date_clinic, max_date_data_clinic],
    min_value=min_date_data_clinic,
    max_value=max_date_data_clinic,
    key="clinic_dashboard_date_range_selector_v5_final", # Incremented key
    help="This date range applies to most charts and Key Performance Indicators (KPIs) unless specified otherwise."
)

# --- Filter dataframes based on selected date range ---
filtered_health_df_clinic = pd.DataFrame(columns=health_df_clinic_main.columns)
if selected_start_date_cl and selected_end_date_cl and 'date' in health_df_clinic_main.columns and not health_df_clinic_main.empty:
    # Ensure 'date' column is datetime64[ns] before attempting .dt accessor
    if not pd.api.types.is_datetime64_ns_dtype(health_df_clinic_main['date']):
         health_df_clinic_main['date'] = pd.to_datetime(health_df_clinic_main['date'], errors='coerce')
    
    # Create 'date_obj' for comparison if 'date' is valid datetime and 'date_obj' doesn't exist or needs refresh
    if pd.api.types.is_datetime64_ns_dtype(health_df_clinic_main['date']): # Re-check after potential conversion
        # Only create/recreate 'date_obj' if it's not there or to ensure it's fresh
        if 'date_obj' not in health_df_clinic_main.columns or not hasattr(health_df_clinic_main['date_obj'].iloc[0] if not health_df_clinic_main.empty else None, 'year'):
            health_df_clinic_main['date_obj'] = health_df_clinic_main['date'].dt.date 
        
        # Now filter using 'date_obj', ensuring it's not NaT
        valid_date_obj_mask = health_df_clinic_main['date_obj'].notna()
        date_range_mask = (health_df_clinic_main['date_obj'] >= selected_start_date_cl) & (health_df_clinic_main['date_obj'] <= selected_end_date_cl)
        filtered_health_df_clinic = health_df_clinic_main[valid_date_obj_mask & date_range_mask].copy()


filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic_main.columns)
if selected_start_date_cl and selected_end_date_cl and 'timestamp' in iot_df_clinic_main.columns and not iot_df_clinic_main.empty:
    if not pd.api.types.is_datetime64_ns_dtype(iot_df_clinic_main['timestamp']):
        iot_df_clinic_main['timestamp'] = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    
    if pd.api.types.is_datetime64_ns_dtype(iot_df_clinic_main['timestamp']):
        if 'date_obj' not in iot_df_clinic_main.columns or not hasattr(iot_df_clinic_main['date_obj'].iloc[0] if not iot_df_clinic_main.empty else None, 'year'):
             iot_df_clinic_main['date_obj'] = iot_df_clinic_main['timestamp'].dt.date
        
        valid_date_obj_mask_iot = iot_df_clinic_main['date_obj'].notna()
        date_range_mask_iot = (iot_df_clinic_main['date_obj'] >= selected_start_date_cl) & (iot_df_clinic_main['date_obj'] <= selected_end_date_cl)
        filtered_iot_df_clinic = iot_df_clinic_main[valid_date_obj_mask_iot & date_range_mask_iot].copy()


# --- Display KPIs ---
date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})" if selected_start_date_cl and selected_end_date_cl else "(Date range not fully set)"

st.subheader(f"Key Disease Service Metrics {date_range_display_str}")
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic)

kpi_cols_clinic_services = st.columns(5)
with kpi_cols_clinic_services[0]:
    tb_pos_rate = clinic_service_kpis.get('tb_sputum_positivity', 0.0)
    render_kpi_card("TB Positivity Rate", f"{tb_pos_rate:.1f}%", "🔬",
                    status="High" if tb_pos_rate > 15 else ("Moderate" if tb_pos_rate > 5 else "Low"),
                    help_text="Percentage of sputum/GeneXpert tests positive for TB in the selected period.")
with kpi_cols_clinic_services[1]:
    mal_pos_rate = clinic_service_kpis.get('malaria_positivity', 0.0)
    render_kpi_card("Malaria Positivity", f"{mal_pos_rate:.1f}%", "🦟",
                    status="High" if mal_pos_rate > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Low",
                    help_text=f"Malaria test (RDT/Microscopy) positivity rate. Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%.")
with kpi_cols_clinic_services[2]:
    avg_tat = clinic_service_kpis.get('avg_test_turnaround_all_tests', 0.0)
    render_kpi_card("Avg. Test TAT", f"{avg_tat:.1f} days", "⏱️",
                    status="High" if avg_tat > app_config.TARGET_TEST_TURNAROUND_DAYS + 1 else ("Moderate" if avg_tat > app_config.TARGET_TEST_TURNAROUND_DAYS else "Low"),
                    help_text=f"Average turnaround time for all conclusive tests. Target: ≤{app_config.TARGET_TEST_TURNAROUND_DAYS} days.")
with kpi_cols_clinic_services[3]:
    hiv_tests_count = clinic_service_kpis.get('hiv_tests_conclusive_period', 0)
    render_kpi_card("HIV Tests Conducted", str(hiv_tests_count), "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='30' alt='HIV'>", icon_is_html=True,
                    status="Low" if hiv_tests_count < 20 else "Moderate", 
                    help_text="Number of unique patients with conclusive HIV test results in the period.")
with kpi_cols_clinic_services[4]:
    drug_stockouts_count = clinic_service_kpis.get('key_drug_stockouts', 0)
    render_kpi_card("Key Drug Stockouts", str(drug_stockouts_count), "💊",
                    status="High" if drug_stockouts_count > 0 else "Low",
                    help_text=f"Number of key disease drugs with < {app_config.CRITICAL_SUPPLY_DAYS} days of supply remaining.")

if not filtered_iot_df_clinic.empty:
    st.subheader(f"Clinic Environment Snapshot {date_range_display_str}")
    clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic)

    kpi_cols_clinic_environment = st.columns(4)
    with kpi_cols_clinic_environment[0]:
        avg_co2_val = clinic_env_kpis.get('avg_co2_overall', 0.0)
        co2_alert_rooms_val = clinic_env_kpis.get('rooms_co2_alert_latest', 0)
        render_kpi_card("Avg. CO2 (All Rooms)", f"{avg_co2_val:.0f} ppm", "💨",
                        status="High" if co2_alert_rooms_val > 0 else "Low",
                        help_text=f"Period average CO2. {co2_alert_rooms_val} room(s) currently > {app_config.CO2_LEVEL_ALERT_PPM}ppm.")
    with kpi_cols_clinic_environment[1]:
        avg_pm25_val = clinic_env_kpis.get('avg_pm25_overall', 0.0)
        pm25_alert_rooms_val = clinic_env_kpis.get('rooms_pm25_alert_latest', 0)
        render_kpi_card("Avg. PM2.5 (All Rooms)", f"{avg_pm25_val:.1f} µg/m³", "🌫️",
                        status="High" if pm25_alert_rooms_val > 0 else "Low",
                        help_text=f"Period average PM2.5. {pm25_alert_rooms_val} room(s) currently > {app_config.PM25_ALERT_UGM3}µg/m³.")
    with kpi_cols_clinic_environment[2]:
        avg_occupancy_val = clinic_env_kpis.get('avg_occupancy_overall', 0.0)
        occupancy_alert_val = clinic_env_kpis.get('high_occupancy_alert_latest', False)
        render_kpi_card("Avg. Occupancy", f"{avg_occupancy_val:.1f} persons", "👨‍👩‍👧‍👦",
                        status="High" if occupancy_alert_val else "Low",
                        help_text=f"Average waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}. Alert if any room's latest reading is high.")
    with kpi_cols_clinic_environment[3]:
        avg_noise_alert_rooms_val = clinic_env_kpis.get('rooms_noise_alert_latest', 0)
        render_kpi_card("Noise Alerts", str(avg_noise_alert_rooms_val), "🔊",
                        status="High" if avg_noise_alert_rooms_val > 0 else "Low",
                        help_text=f"Rooms with latest noise levels > {app_config.NOISE_LEVEL_ALERT_DB}dB.")
st.markdown("---")

tab_titles_clinic = [
    "🔬 Disease Testing Insights", "💊 Supply Chain Management",
    "🧍 Patient Focus & Alerts", "🌿 Clinic Environment Details"
]
tab_tests, tab_supplies, tab_patients, tab_environment = st.tabs(tab_titles_clinic)

with tab_tests:
    st.subheader("Disease-Specific Testing Performance and Trends")
    if not filtered_health_df_clinic.empty and all(c in filtered_health_df_clinic.columns for c in ['test_type', 'test_result', 'patient_id', 'date']):
        col_test_dist, col_test_tat_trend = st.columns([0.45, 0.55])
        with col_test_dist:
            st.markdown("###### **TB Test Result Distribution**")
            tb_tests_df_viz = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].str.contains("Sputum|GeneXpert", case=False, na=False)].copy()
            if not tb_tests_df_viz.empty:
                tb_results_dist_summary = tb_tests_df_viz.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
                tb_results_dist_summary.columns = ['Test Result', 'Unique Patients']
                tb_results_dist_summary_conclusive = tb_results_dist_summary[~tb_results_dist_summary['Test Result'].isin(['Unknown', 'N/A', 'nan', 'Pending'])]
                if not tb_results_dist_summary_conclusive.empty:
                    st.plotly_chart(plot_donut_chart(tb_results_dist_summary_conclusive, 'Test Result', 'Unique Patients',
                                                    title="TB Test Results (Conclusive)", height=app_config.COMPACT_PLOT_HEIGHT + 20,
                                                    color_discrete_map={"Positive": app_config.RISK_STATUS_COLORS["High"], "Negative": app_config.RISK_STATUS_COLORS["Low"]}),
                                   use_container_width=True)
                else: st.caption("No conclusive TB test result data for donut chart in the selected period.")
            else: st.caption("No TB tests found in the selected period to display distribution.")

        with col_test_tat_trend:
            st.markdown("###### **Daily Average Test Turnaround Time (TAT)**")
            if 'test_turnaround_days' in filtered_health_df_clinic.columns and 'date' in filtered_health_df_clinic.columns: 
                tat_trend_data_src = filtered_health_df_clinic[
                    filtered_health_df_clinic['test_turnaround_days'].notna() &
                    (~filtered_health_df_clinic['test_result'].isin(['Pending', 'N/A', 'Unknown']))
                ].copy()
                if not tat_trend_data_src.empty:
                    daily_avg_tat_trend = get_trend_data(tat_trend_data_src,'test_turnaround_days', period='D', date_col='date', agg_func='mean')
                    if not daily_avg_tat_trend.empty:
                        st.plotly_chart(plot_annotated_line_chart(
                            daily_avg_tat_trend, "Daily Avg. Test Turnaround Time", y_axis_title="Days",
                            target_line=app_config.TARGET_TEST_TURNAROUND_DAYS, target_label="Target TAT",
                            height=app_config.COMPACT_PLOT_HEIGHT + 20, show_anomalies=True, date_format="%d %b"
                        ), use_container_width=True)
                    else: st.caption("No aggregated TAT data available for trend in the selected period.")
                else: st.caption("No conclusive tests with TAT data found in the selected period for trend.")
            else: st.caption("Test Turnaround Time data ('test_turnaround_days') or result date ('date') missing for TAT trend analysis.")
    else:
        st.info("Health records are empty or missing key columns for the Disease Testing Insights tab.")


with tab_supplies:
    st.subheader("Medical Supply Levels & Consumption Forecast")
    if health_df_clinic_main is not None and not health_df_clinic_main.empty and \
       all(c in health_df_clinic_main.columns for c in ['item', 'date', 'stock_on_hand', 'consumption_rate_per_day']):

        supply_forecast_df = get_supply_forecast_data(health_df_clinic_main, forecast_days_out=28)

        if supply_forecast_df is not None and not supply_forecast_df.empty:
            key_drug_items_for_select = sorted([
                item for item in supply_forecast_df['item'].unique()
                if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)
            ])

            if not key_drug_items_for_select:
                st.info("No forecast data available for the defined key disease drugs.")
            else:
                selected_drug_for_forecast = st.selectbox(
                    "Select Key Drug for Forecast Details:", key_drug_items_for_select,
                    key="clinic_supply_item_forecast_selector_final_v2", # Incremented key
                    help="View the forecasted days of supply remaining for the selected drug."
                )
                if selected_drug_for_forecast:
                    item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
                    if not item_specific_forecast_df.empty:
                        item_specific_forecast_df.sort_values('date', inplace=True)
                        
                        current_day_info = item_specific_forecast_df.iloc[0] 
                        forecast_plot_title = (
                            f"Forecast: {selected_drug_for_forecast}<br>"
                            f"<sup_>Current Stock: {current_day_info.get('current_stock',0):.0f} units | "
                            f"Est. Daily Use: {current_day_info.get('consumption_rate',0):.1f} units/day | "
                            f"Est. Stockout: {pd.to_datetime(current_day_info.get('estimated_stockout_date')).strftime('%d %b %Y') if pd.notna(current_day_info.get('estimated_stockout_date')) else 'N/A'}</sup>"
                        )
                        
                        st.plotly_chart(plot_annotated_line_chart(
                            item_specific_forecast_df.set_index('date')['forecast_days'],
                            title=forecast_plot_title,
                            y_axis_title="Days of Supply Remaining",
                            target_line=app_config.CRITICAL_SUPPLY_DAYS,
                            target_label=f"Critical Level ({app_config.CRITICAL_SUPPLY_DAYS} Days)",
                            show_ci=True,
                            lower_bound_series=item_specific_forecast_df.set_index('date')['lower_ci'],
                            upper_bound_series=item_specific_forecast_df.set_index('date')['upper_ci'],
                            height=app_config.DEFAULT_PLOT_HEIGHT + 60,
                            show_anomalies=False
                        ), use_container_width=True)
                    else: st.info(f"No forecast data found for the selected drug: {selected_drug_for_forecast}.")
        else:
            st.warning("Supply forecast data could not be generated. This may be due to missing consumption rates or stock levels.")
    else:
        st.error("CRITICAL FOR SUPPLY TAB: Cannot generate supply forecasts. Base health data is unusable or missing essential columns (item, date, stock_on_hand, consumption_rate_per_day).")


with tab_patients:
    st.subheader("Patient Load & High-Risk Case Identification")
    if not filtered_health_df_clinic.empty and all(c in filtered_health_df_clinic.columns for c in ['condition', 'date', 'patient_id']):
        conditions_for_load_chart = app_config.KEY_CONDITIONS_FOR_TRENDS
        patient_load_df_src = filtered_health_df_clinic[
            filtered_health_df_clinic['condition'].isin(conditions_for_load_chart) &
            (filtered_health_df_clinic['patient_id'] != 'Unknown')
        ].copy()

        if not patient_load_df_src.empty:
            daily_patient_load_summary = patient_load_df_src.groupby(
                [pd.Grouper(key='date', freq='D'), 'condition']
            )['patient_id'].nunique().reset_index()
            daily_patient_load_summary.rename(columns={'patient_id': 'unique_patients'}, inplace=True)

            if not daily_patient_load_summary.empty:
                st.plotly_chart(plot_bar_chart(
                    daily_patient_load_summary, x_col='date', y_col='unique_patients',
                    title="Daily Unique Patient Visits by Key Condition", color_col='condition',
                    barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT + 70,
                    y_axis_title="Unique Patients per Day", x_axis_title="Date",
                    color_discrete_map=app_config.DISEASE_COLORS,
                    text_auto=False 
                ), use_container_width=True)
            else: st.caption("No patient load data for key conditions found in the selected period to display the chart.")
        else: st.caption("No patients with key conditions recorded in the selected period.")
    else:
        st.info("Health records (filtered for period) are empty or missing key columns for Patient Load chart display.")

    st.markdown("---"); st.markdown("###### **Flagged Patient Cases for Clinical Review (Selected Period)**")
    if not filtered_health_df_clinic.empty:
        flagged_patients_clinic_review_df = get_patient_alerts_for_clinic(
            filtered_health_df_clinic,
            risk_threshold_moderate=app_config.RISK_THRESHOLDS['moderate']
        )
        if flagged_patients_clinic_review_df is not None and not flagged_patients_clinic_review_df.empty:
            st.markdown(f"Found **{len(flagged_patients_clinic_review_df)}** patient cases flagged for review based on high risk, recent critical positive tests, or overdue critical tests within the period.")
            
            cols_for_alert_table_clinic = ['patient_id', 'condition', 'ai_risk_score', 'alert_reason',
                                           'test_result', 'test_type', 'hiv_viral_load', 'priority_score', 'date']
            alerts_display_df_clinic = flagged_patients_clinic_review_df[
                [col for col in cols_for_alert_table_clinic if col in flagged_patients_clinic_review_df.columns]
            ].copy()
            
            st.dataframe(
                alerts_display_df_clinic.head(25),
                use_container_width=True,
                column_config={
                    "ai_risk_score": st.column_config.ProgressColumn("AI Risk", format="%d", min_value=0, max_value=100, width="medium"),
                    "date": st.column_config.DateColumn("Latest Record Date", format="YYYY-MM-DD"),
                    "alert_reason": st.column_config.TextColumn("Alert Reason(s)", width="large", help="Reasons why this patient case is flagged."),
                    "priority_score": st.column_config.NumberColumn("Priority", help="Calculated alert priority (higher is more urgent).", format="%d"),
                    "hiv_viral_load": st.column_config.NumberColumn("HIV VL", format="%.0f copies/mL", help="HIV Viral Load if applicable.") #Changed format to %.0f
                },
                height=450, hide_index=True
            )
        else: st.info("No specific patient cases flagged for clinical review in the selected period based on current criteria.")
    else: st.info("Filtered health records are empty; cannot generate patient alerts for clinical review.")


with tab_environment:
    st.subheader("Clinic Environmental Monitoring - Trends & Details")
    if not filtered_iot_df_clinic.empty and 'timestamp' in filtered_iot_df_clinic.columns:
        env_summary_for_tab = get_clinic_environmental_summary(filtered_iot_df_clinic)
        st.markdown(f"""
        **Current Environmental Alerts (based on latest readings in period):**
        - **CO2 Alerts:** {env_summary_for_tab.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.
        - **PM2.5 Alerts:** {env_summary_for_tab.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.
        - **Noise Alerts:** {env_summary_for_tab.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB.
        """)
        if env_summary_for_tab.get('high_occupancy_alert_latest', False):
            st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** At least one area has occupancy > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons (latest reading). Consider patient flow adjustments.")

        env_trend_plot_cols = st.columns(2)
        with env_trend_plot_cols[0]:
            if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
                hourly_avg_co2_trend = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_co2_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        hourly_avg_co2_trend, "Hourly Avg. CO2 Levels (All Rooms)", y_axis_title="CO2 (ppm)",
                        target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold",
                        height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"
                    ), use_container_width=True)
                else: st.caption("No CO2 trend data available for this period.")
            else: st.caption("CO2 data ('avg_co2_ppm') is missing from IoT records for trend analysis.")

        with env_trend_plot_cols[1]:
            if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
                hourly_avg_occupancy_trend = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_occupancy_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        hourly_avg_occupancy_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons",
                        target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy",
                        height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"
                    ), use_container_width=True)
                else: st.caption("No occupancy trend data available for this period.")
            else: st.caption("Occupancy data ('waiting_room_occupancy') missing for trend analysis.")

        st.markdown("---")
        st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
        latest_room_cols_display = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy']
        available_latest_cols = [col for col in latest_room_cols_display if col in filtered_iot_df_clinic.columns]
        
        if all(c in available_latest_cols for c in ['timestamp', 'clinic_id', 'room_name']): # Ensure base columns exist
            if not filtered_iot_df_clinic.empty: # Ensure there's data to sort/drop from
                latest_room_sensor_readings = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
                if not latest_room_sensor_readings.empty:
                    st.dataframe(
                        latest_room_sensor_readings[available_latest_cols].tail(15),
                        use_container_width=True, height=380,
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn("Last Reading At", format="YYYY-MM-DD HH:mm"),
                            "avg_co2_ppm": st.column_config.NumberColumn("CO2 (ppm)", format="%d ppm"),
                            "avg_pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.1f µg/m³"),
                            "avg_temp_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.1f°C"),
                            "avg_humidity_rh": st.column_config.NumberColumn("Humidity (%RH)", format="%d%%"),
                            "avg_noise_db": st.column_config.NumberColumn("Noise Level (dB)", format="%d dB"),
                            "waiting_room_occupancy": st.column_config.NumberColumn("Occupancy", format="%d persons"),
                        },
                        hide_index=True
                    )
                else: st.caption("No detailed room sensor readings available for the end of the selected period after filtering.")
            else:  st.caption("IoT data for the selected period is empty. Cannot display latest room readings.")
        else:
            st.caption(f"Essential columns ('timestamp', 'clinic_id', 'room_name') missing for detailed room readings display. Available: {', '.join(available_latest_cols)}")
    else:
        st.info("No clinic environmental data available for this tab (either no data loaded, or none within the selected date range), or 'timestamp' column is problematic.")
