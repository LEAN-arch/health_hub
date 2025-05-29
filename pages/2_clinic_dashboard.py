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

if health_df.empty and iot_df_clinic.empty :
    st.error("🚨 CRITICAL Error: Could not load health records or IoT data. Clinic Dashboard cannot be displayed.")
else:
    st.title("🏥 Clinic Operations & Environmental Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility Environment**")
    st.markdown("---")

    st.sidebar.header("Clinic Filters")
    min_date_cl_page = health_df['date'].min().date() if not health_df.empty and 'date' in health_df.columns and health_df['date'].notna().any() else pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND)
    max_date_cl_page = health_df['date'].max().date() if not health_df.empty and 'date' in health_df.columns and health_df['date'].notna().any() else pd.Timestamp('today').date()
    if min_date_cl_page > max_date_cl_page: min_date_cl_page = max_date_cl_page
    
    default_start_cl = max_date_cl_page - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1)
    if default_start_cl < min_date_cl_page: default_start_cl = min_date_cl_page
    if default_start_cl > max_date_cl_page and min_date_cl_page <= max_date_cl_page: default_start_cl = min_date_cl_page
    elif default_start_cl > max_date_cl_page : default_start_cl = max_date_cl_page
    
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range for Analysis", [default_start_cl, max_date_cl_page],
        min_value=min_date_cl_page, max_value=max_date_cl_page, key="clinic_date_range_final_v2", # Unique key
        help="Applies to most charts and KPIs unless specified."
    )
    
    filtered_df = pd.DataFrame()
    if start_date and end_date and start_date <= end_date and not health_df.empty:
        filtered_df = health_df[(health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)].copy()
    elif not health_df.empty: filtered_df = health_df.copy()

    filtered_iot_df_clinic = pd.DataFrame()
    if start_date and end_date and start_date <= end_date and not iot_df_clinic.empty:
        filtered_iot_df_clinic = iot_df_clinic[(iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)].copy()
    elif not iot_df_clinic.empty: filtered_iot_df_clinic = iot_df_clinic.copy()

    clinic_kpis = get_clinic_summary(filtered_df) if not filtered_df.empty else {} # Get empty dict if no data
    st.subheader(f"Key Disease Service Metrics ({start_date.strftime('%d %b')} - {end_date.strftime('%d %b %Y') if end_date else 'N/A'})")
    kpi_cols_disease_cl = st.columns(5)
    with kpi_cols_disease_cl[0]: render_kpi_card("TB Sputum Positivity", f"{clinic_kpis.get('tb_sputum_positivity',0):.1f}%", "🔬", status="High" if clinic_kpis.get('tb_sputum_positivity',0) > 10 else "Moderate")
    with kpi_cols_disease_cl[1]: render_kpi_card("Malaria RDT Positivity", f"{clinic_kpis.get('malaria_rdt_positivity',0):.1f}%", "🦟", status="High" if clinic_kpis.get('malaria_rdt_positivity',0) > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Moderate")
    with kpi_cols_disease_cl[2]: render_kpi_card("STI Tests Pending", str(clinic_kpis.get('sti_tests_pending',0)), "🧪", status="Moderate" if clinic_kpis.get('sti_tests_pending',0) > 5 else "Low")
    with kpi_cols_disease_cl[3]: 
        hiv_icon = "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='28' alt='HIV' style='vertical-align: middle;'>"
        render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), hiv_icon, icon_is_html=True, help_text="Number of conclusive HIV tests conducted.")
    with kpi_cols_disease_cl[4]: render_kpi_card("Key Drug Stockouts", str(clinic_kpis.get('critical_disease_supply_items',0)), "💊", status="High" if clinic_kpis.get('critical_disease_supply_items',0) > 0 else "Low")

    st.subheader(f"Clinic Environment Snapshot (Data from {start_date.strftime('%d %b')} - {end_date.strftime('%d %b %Y') if end_date else 'N/A'})")
    env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic) if not filtered_iot_df_clinic.empty else {}
    kpi_cols_env_cl = st.columns(4)
    with kpi_cols_env_cl[0]: render_kpi_card("Avg. CO2", f"{env_summary_clinic.get('avg_co2',0):.0f} ppm", "💨", status="High" if env_summary_clinic.get('co2_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {env_summary_clinic.get('co2_alert_rooms',0)}")
    with kpi_cols_env_cl[1]: render_kpi_card("Avg. PM2.5", f"{env_summary_clinic.get('avg_pm25',0):.1f} µg/m³", "🌫️", status="High" if env_summary_clinic.get('pm25_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {env_summary_clinic.get('pm25_alert_rooms',0)}")
    with kpi_cols_env_cl[2]: render_kpi_card("Avg. Occupancy", f"{env_summary_clinic.get('avg_occupancy',0):.1f}", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
    with kpi_cols_env_cl[3]: render_kpi_card("Sanitizer Use", f"{env_summary_clinic.get('avg_sanitizer_use_hr',0):.1f}/hr", "🧴", status="Low" if env_summary_clinic.get('avg_sanitizer_use_hr',0) < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High")
    st.markdown("---")

    tab_tests_cl, tab_supplies_cl, tab_patients_cl, tab_environment_cl = st.tabs(["🔬 Disease Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Clinic Environment"])

    with tab_tests_cl:
        # ... (Disease testing donut charts and turnaround trend as before) ...
        st.subheader("Test Results & Performance")
        if not filtered_df.empty and 'test_type' in filtered_df.columns and 'test_result' in filtered_df.columns:
            col_test1_cl, col_test2_cl = st.columns([0.4, 0.6]) 
            with col_test1_cl:
                tb_tests_df = filtered_df[filtered_df['test_type'].astype(str).str.contains("Sputum|GeneXpert", case=False, na=False)].copy()
                tb_results_summary = tb_tests_df.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
                tb_results_summary.columns = ['Test Result', 'Count']; tb_results_summary = tb_results_summary[~tb_results_summary['Test Result'].isin(['Unknown', 'N/A', 'nan', 'Pending'])]
                if not tb_results_summary.empty: st.plotly_chart(plot_donut_chart(tb_results_summary, 'Test Result', 'Count', "TB Test Results"), use_container_width=True)
                else: st.caption("No TB test result data.")
            with col_test2_cl:
                turnaround_trend_df_cl = filtered_df.dropna(subset=['test_turnaround_days', 'test_date']); turnaround_trend_df_cl['test_date'] = pd.to_datetime(turnaround_trend_df_cl['test_date'], errors='coerce'); turnaround_trend_df_cl.dropna(subset=['test_date'], inplace=True)
                if not turnaround_trend_df_cl.empty:
                    turnaround_trend_cl = get_trend_data(turnaround_trend_df_cl,'test_turnaround_days', period='D', date_col='test_date')
                    if not turnaround_trend_cl.empty: st.plotly_chart(plot_annotated_line_chart(turnaround_trend_cl, "Avg. Test Turnaround Time", y_axis_title="Days", target_line=app_config.TARGET_TEST_TURNAROUND_DAYS), use_container_width=True)
                    else: st.caption("No aggregated turnaround time data for trend.")
                else: st.caption("No raw turnaround time data for trend.")
        else: st.caption("Test data columns missing for disease testing analysis.")


    with tab_supplies_cl:
        # ... (Supply forecast for key disease drugs as before) ...
        st.subheader("Supply Levels & Forecast (Key Disease Drugs)"); supply_forecast_df_all_items = get_supply_forecast_data(health_df) 
        if not supply_forecast_df_all_items.empty:
            key_drug_substrings = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Benznidazole', 'ORS', 'Oxygen', 'Metronidazole']
            available_key_items_cl = sorted([item for item in supply_forecast_df_all_items['item'].unique() if pd.notna(item) and any(sub.lower() in str(item).lower() for sub in key_drug_substrings)])
            if not available_key_items_cl: st.info("No key disease drug forecast data.")
            else:
                selected_item_clinic_tab = st.selectbox("Select Key Drug for Forecast:", available_key_items_cl, key="supply_item_select_clinic_tab_final")
                if selected_item_clinic_tab: 
                    item_forecast_df_cl = supply_forecast_df_all_items[supply_forecast_df_all_items['item'] == selected_item_clinic_tab].copy(); item_forecast_df_cl.sort_values('date', inplace=True); item_forecast_df_cl.set_index('date', inplace=True)
                    if not item_forecast_df_cl.empty: st.plotly_chart(plot_annotated_line_chart(item_forecast_df_cl['forecast_days'],f"Forecast: {selected_item_clinic_tab}",y_axis_title="Days of Supply",target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)",show_ci=True, lower_bound_series=item_forecast_df_cl['lower_ci'], upper_bound_series=item_forecast_df_cl['upper_ci'],height=app_config.DEFAULT_PLOT_HEIGHT + 50), use_container_width=True)
        else: st.caption("No supply data for forecasts.")

    with tab_patients_cl:
        # ... (Patient load by KEY_CONDITIONS and flagged patient list as before) ...
        st.subheader("Patient Load by Key Conditions");
        if not filtered_df.empty and 'condition' in filtered_df.columns:
            key_conditions_clinic_df = filtered_df[filtered_df['condition'].astype(str).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)]
            if not key_conditions_clinic_df.empty and 'date' in key_conditions_clinic_df.columns:
                patient_condition_summary = key_conditions_clinic_df.groupby([pd.Grouper(key='date', freq='D'), 'condition'])['patient_id'].nunique().reset_index()
                patient_condition_summary.rename(columns={'patient_id': 'patient_count'}, inplace=True)
                if not patient_condition_summary.empty:
                    patient_pivot = patient_condition_summary.pivot_table(index='date', columns='condition', values='patient_count', fill_value=0).reset_index()
                    patient_melt = patient_pivot.melt(id_vars='date', var_name='condition', value_name='patient_count')
                    st.plotly_chart(plot_bar_chart(patient_melt, x_col='date', y_col='patient_count', title="Daily Patient Count by Key Condition", color_col='condition', barmode='stack',height=app_config.DEFAULT_PLOT_HEIGHT+50), use_container_width=True)
                else: st.caption("No patient data for key conditions in selected period.")
            else: st.caption("No patients with key conditions or 'date' column issue in selected period.")
        else: st.caption("No patient data or 'condition' column loaded.")
        st.markdown("---"); st.markdown("###### Flagged Patient Cases for Review (Key Diseases)")
        flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS['moderate']) if not filtered_df.empty else pd.DataFrame()
        if not flagged_patients_clinic_df.empty: st.dataframe(flagged_patients_clinic_df[['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result']].sort_values(by='ai_risk_score', ascending=False).head(10), use_container_width=True, column_config={ "ai_risk_score": st.column_config.NumberColumn(format="%d")})
        else: st.info("No specific patient cases flagged for review.")

    with tab_environment_cl:
        # ... (Environmental KPIs, trends, and detailed room readings logic as in the previous "act as SME..." response) ...
        st.subheader("Real-time Environmental Monitoring")
        if not filtered_iot_df_clinic.empty:
            # env_summary_clinic already calculated at top of page
            st.markdown(f"**Alerts (based on latest room readings):** {env_summary_clinic.get('co2_alert_rooms',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm; {env_summary_clinic.get('pm25_alert_rooms',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.")
            if env_summary_clinic.get('high_occupancy_alert'): st.warning(f"High waiting room occupancy detected (above {app_config.TARGET_WAITING_ROOM_OCCUPANCY}).")
            
            env_trend_cols_cl = st.columns(2)
            with env_trend_cols_cl[0]:
                co2_trend_cl = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not co2_trend_cl.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend_cl, "Hourly Avg. CO2 Levels", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
                else: st.caption("No CO2 trend.")
            with env_trend_cols_cl[1]:
                occupancy_trend_cl = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not occupancy_trend_cl.empty: st.plotly_chart(plot_annotated_line_chart(occupancy_trend_cl, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
                else: st.caption("No occupancy trend.")
            
            st.subheader("Latest Room Readings")
            latest_room_readings_cl = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last') if 'timestamp' in filtered_iot_df_clinic and 'clinic_id' in filtered_iot_df_clinic and 'room_name' in filtered_iot_df_clinic else pd.DataFrame()
            if not latest_room_readings_cl.empty: st.dataframe(latest_room_readings_cl[['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'waiting_room_occupancy']].tail(), use_container_width=True, height=250)
            else: st.caption("No detailed room readings for period.")
        else: st.info("No clinic environmental data for selected period.")
