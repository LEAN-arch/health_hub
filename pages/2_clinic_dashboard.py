# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data, # New
    get_clinic_summary, # Updated for disease focus
    get_clinic_environmental_summary, # New
    get_trend_data,
    get_supply_forecast_data,
    get_patient_alerts_for_clinic
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_donut_chart,
    plot_annotated_line_chart,
    plot_bar_chart
)
import logging

st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"CSS file not found at {app_config.STYLE_CSS}.")
load_css()

@st.cache_data(ttl=3600)
def get_clinic_page_data_extended():
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    return health_df, iot_df
health_df, iot_df_clinic = get_clinic_page_data_extended()

if health_df.empty and iot_df_clinic.empty : # Check if both are problematic
    st.error("🚨 CRITICAL Error: Could not load health records or IoT data. Clinic Dashboard cannot be displayed.")
else:
    st.title("🏥 Clinic Operations & Environmental Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility Environment**")
    st.markdown("---")

    st.sidebar.header("Clinic Filters")
    # ... (Date filter logic as in previous complete file, ensuring robustness) ...
    min_date_cl = health_df['date'].min().date() if not health_df.empty and 'date' in health_df else pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND)
    max_date_cl = health_df['date'].max().date() if not health_df.empty and 'date' in health_df else pd.Timestamp('today').date()
    default_start_cl = max_date_cl - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1)
    if default_start_cl < min_date_cl: default_start_cl = min_date_cl
    if default_start_cl > max_date_cl and min_date_cl <= max_date_cl : default_start_cl = min_date_cl 
    elif default_start_cl > max_date_cl : default_start_cl = max_date_cl

    start_date, end_date = st.sidebar.date_input(
        "Select Date Range for Analysis", [default_start_cl, max_date_cl],
        min_value=min_date_cl, max_value=max_date_cl, key="clinic_date_range_final",
        help="Applies to most charts and KPIs unless specified."
    )
    
    filtered_df = pd.DataFrame()
    if start_date and end_date and start_date <= end_date and not health_df.empty:
        filtered_df = health_df[(health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)].copy()
    elif not health_df.empty: filtered_df = health_df.copy() # Fallback

    filtered_iot_df_clinic = pd.DataFrame()
    if start_date and end_date and start_date <= end_date and not iot_df_clinic.empty:
        filtered_iot_df_clinic = iot_df_clinic[(iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)].copy()
    elif not iot_df_clinic.empty: filtered_iot_df_clinic = iot_df_clinic.copy()


    # --- KPIs ---
    clinic_kpis = get_clinic_summary(filtered_df) # Uses disease-focused logic
    st.subheader(f"Key Disease Service Metrics ({start_date.strftime('%d %b')} - {end_date.strftime('%d %b %Y') if end_date else 'N/A'})")
    kpi_cols_disease = st.columns(5) # Now 5 for the new focused KPIs
    with kpi_cols_disease[0]: render_kpi_card("TB Sputum Positivity", f"{clinic_kpis.get('tb_sputum_positivity',0):.1f}%", "🔬", status="High" if clinic_kpis.get('tb_sputum_positivity',0) > 10 else "Moderate")
    with kpi_cols_disease[1]: render_kpi_card("Malaria RDT Positivity", f"{clinic_kpis.get('malaria_rdt_positivity',0):.1f}%", "🦟", status="High" if clinic_kpis.get('malaria_rdt_positivity',0) > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Moderate")
    with kpi_cols_disease[2]: render_kpi_card("STI Tests Pending", str(clinic_kpis.get('sti_tests_pending',0)), "🧪", status="Moderate" if clinic_kpis.get('sti_tests_pending',0) > 5 else "Low")
    with kpi_cols_disease[3]: render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='24'>", help_text="Number of HIV tests conducted.")
    with kpi_cols_disease[4]: render_kpi_card("Key Drug Stockouts", str(clinic_kpis.get('critical_disease_supply_items',0)), "💊", status="High" if clinic_kpis.get('critical_disease_supply_items',0) > 0 else "Low", help_text="Number of critical disease drugs with low stock.")

    st.subheader(f"Clinic Environment Snapshot (Latest Data in Range)")
    env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic)
    kpi_cols_env = st.columns(4)
    with kpi_cols_env[0]: render_kpi_card("Avg. CO2", f"{env_summary_clinic.get('avg_co2',0):.0f} ppm", "💨", status="High" if env_summary_clinic.get('co2_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {env_summary_clinic.get('co2_alert_rooms',0)}")
    with kpi_cols_env[1]: render_kpi_card("Avg. PM2.5", f"{env_summary_clinic.get('avg_pm25',0):.1f} µg/m³", "🌫️", status="High" if env_summary_clinic.get('pm25_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {env_summary_clinic.get('pm25_alert_rooms',0)}")
    with kpi_cols_env[2]: render_kpi_card("Avg. Occupancy", f"{env_summary_clinic.get('avg_occupancy',0):.1f}", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
    with kpi_cols_env[3]: render_kpi_card("Sanitizer Use", f"{env_summary_clinic.get('avg_sanitizer_use_hr',0):.1f}/hr", "🧴", status="Low" if env_summary_clinic.get('avg_sanitizer_use_hr',0) < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High", help_text="Avg. dispenses/hr/unit.")
    st.markdown("---")

    tab_tests, tab_supplies, tab_patients, tab_environment = st.tabs(["🔬 Disease Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Clinic Environment"])

    with tab_tests:
        st.subheader("Disease-Specific Test Results & Performance")
        # ... (Plot donut charts for TB, Malaria, HIV, STI test results if data allows) ...
        # Example for TB
        if 'test_type' in filtered_df.columns and 'test_result' in filtered_df.columns:
            tb_tests_df = filtered_df[filtered_df['test_type'].astype(str).str.contains("Sputum", case=False, na=False)].copy()
            tb_results_summary = tb_tests_df.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
            tb_results_summary.columns = ['Test Result', 'Count']
            tb_results_summary = tb_results_summary[~tb_results_summary['Test Result'].isin(['Unknown', 'N/A', 'nan'])]
            if not tb_results_summary.empty:
                st.plotly_chart(plot_donut_chart(tb_results_summary, 'Test Result', 'Count', "TB Test Result Distribution"), use_container_width=True)
            else: st.caption("No TB test result data.")
        else: st.caption("Test data columns missing for TB results.")
        # Add turnaround trend for CRITICAL_TESTS_PENDING
        # ...

    with tab_supplies:
        # ... (Supply forecast logic as before, ensure relevant key disease drugs are selectable) ...
        st.subheader("Supply Levels & Forecast (Key Disease Drugs)")
        supply_forecast_df_all_items = get_supply_forecast_data(health_df) 
        if not supply_forecast_df_all_items.empty:
            key_drug_substrings = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Benznidazole']
            available_key_items = sorted([
                item for item in supply_forecast_df_all_items['item'].unique() 
                if pd.notna(item) and any(sub.lower() in str(item).lower() for sub in key_drug_substrings)
            ])
            if not available_key_items: st.info("No key disease drug forecast data available.")
            else:
                selected_item_clinic = st.selectbox("Select Key Drug for Forecast:", available_key_items, key="supply_item_select_clinic_final")
                if selected_item_clinic: # ... (rest of plotting logic for selected item as before) ...
                    item_forecast_df = supply_forecast_df_all_items[supply_forecast_df_all_items['item'] == selected_item_clinic].copy(); item_forecast_df.sort_values('date', inplace=True); item_forecast_df.set_index('date', inplace=True)
                    if not item_forecast_df.empty: st.plotly_chart(plot_annotated_line_chart(item_forecast_df['forecast_days'],f"Forecast: {selected_item_clinic}",y_axis_title="Days of Supply",target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)",show_ci=True, lower_bound_series=item_forecast_df['lower_ci'], upper_bound_series=item_forecast_df['upper_ci'],height=app_config.DEFAULT_PLOT_HEIGHT + 50), use_container_width=True)
        else: st.caption("No supply data for forecasts.")


    with tab_patients:
        # ... (Patient load by KEY_CONDITIONS_FOR_TRENDS, and flagged patient list logic focusing on these diseases) ...
        st.subheader("Patient Load by Key Conditions")
        if not filtered_df.empty:
            key_conditions_clinic_df = filtered_df[filtered_df['condition'].astype(str).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)]
            if not key_conditions_clinic_df.empty:
                patient_condition_summary = key_conditions_clinic_df.groupby([pd.Grouper(key='date', freq='D'), 'condition'])['patient_id'].nunique().reset_index()
                patient_condition_summary.rename(columns={'patient_id': 'patient_count'}, inplace=True)
                if not patient_condition_summary.empty:
                    patient_pivot = patient_condition_summary.pivot_table(index='date', columns='condition', values='patient_count', fill_value=0).reset_index()
                    patient_melt = patient_pivot.melt(id_vars='date', var_name='condition', value_name='patient_count')
                    st.plotly_chart(plot_bar_chart(patient_melt, x_col='date', y_col='patient_count', title="Daily Patient Count by Key Condition", color_col='condition', barmode='stack'), use_container_width=True)
                else: st.caption("No patient data for key conditions in selected period.")
            else: st.caption("No patients with key conditions in selected period.")
        else: st.caption("No patient data loaded.")
        
        st.markdown("---"); st.markdown("###### Flagged Patient Cases for Review (Key Diseases)")
        flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS['moderate'])
        # Further filter flagged_patients_clinic_df for KEY_CONDITIONS if needed, or ensure get_patient_alerts_for_clinic prioritizes them
        if not flagged_patients_clinic_df.empty:
            st.dataframe(flagged_patients_clinic_df[['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result']].sort_values(by='ai_risk_score', ascending=False).head(10), use_container_width=True)
        else: st.info("No specific patient cases flagged for review.")


    with tab_environment:
        # ... (Environmental KPIs, trends, and detailed room readings logic as in the previous response's Clinic Dashboard) ...
        # This was already well-fleshed out in the previous step.
        st.subheader("Real-time Environmental Monitoring")
        if not filtered_iot_df_clinic.empty:
            # env_summary_clinic already calculated
            st.markdown(f"**Alerts:** {env_summary_clinic.get('co2_alert_rooms',0)} room(s) with CO2 > threshold, {env_summary_clinic.get('pm25_alert_rooms',0)} room(s) with PM2.5 > threshold.")
            env_trend_cols = st.columns(2)
            with env_trend_cols[0]:
                co2_trend = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend, "Hourly Avg. CO2 Levels", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold"), use_container_width=True)
            with env_trend_cols[1]:
                temp_trend = get_trend_data(filtered_iot_df_clinic, 'avg_temp_celsius', date_col='timestamp', period='H', agg_func='mean')
                if not temp_trend.empty: st.plotly_chart(plot_annotated_line_chart(temp_trend, "Hourly Avg. Temperature", y_axis_title="Temp (°C)"), use_container_width=True)
            # Patient Flow and Hand Hygiene from previous example can be added here
            st.markdown("##### Patient Flow & Occupancy") # ... occupancy and throughput metrics/charts
            st.markdown("##### Hand Hygiene (Proxy)") # ... sanitizer trend
            st.subheader("Latest Room Readings") # ... latest_room_readings dataframe
            latest_room_readings = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
            if not latest_room_readings.empty: st.dataframe(latest_room_readings[['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'waiting_room_occupancy']].tail(), use_container_width=True)
            else: st.caption("No detailed room readings.")
        else: st.info("No clinic environmental data for selected period.")
