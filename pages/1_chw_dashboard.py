# pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging # Make sure logging is imported
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_trend_data
)
from utils.ui_visualization_helpers import (
    render_kpi_card, # Using the direct rendering function
    render_traffic_light, # Using the direct rendering function
    plot_annotated_line_chart
)

st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Get logger for this page

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"CSS file not found at {app_config.STYLE_CSS}.")
load_css()

@st.cache_data(ttl=3600)
def get_chw_page_data(): return load_health_records()
health_df = get_chw_page_data()

if health_df.empty:
    st.error("🚨 Critical Error: Could not load health records. CHW Dashboard cannot be displayed.")
else:
    st.title("🧑‍⚕️ Community Health Worker Dashboard")
    st.markdown("**Field Insights, Patient Prioritization & Wellness Monitoring**")
    st.markdown("---")

    st.sidebar.header("CHW Filters")
    min_date_chw_page = health_df['date'].min().date() if not health_df.empty and 'date' in health_df.columns and health_df['date'].notna().any() else pd.Timestamp('today').date() - pd.Timedelta(days=30)
    max_date_chw_page = health_df['date'].max().date() if not health_df.empty and 'date' in health_df.columns and health_df['date'].notna().any() else pd.Timestamp('today').date()
    
    # Ensure min_date is not after max_date if data is sparse
    if min_date_chw_page > max_date_chw_page: min_date_chw_page = max_date_chw_page 

    selected_view_date = st.sidebar.date_input(
        "View Data For Date", max_date_chw_page, 
        min_value=min_date_chw_page, max_value=max_date_chw_page, 
        key="chw_view_date_final_v2", # Unique key
        help="Select date for daily summaries and tasks."
    )
    current_day_df = health_df[health_df['date'].dt.date == selected_view_date].copy() if not health_df.empty and selected_view_date else pd.DataFrame()

    chw_kpis = get_chw_summary(current_day_df)
    
    st.subheader(f"Daily Snapshot: {selected_view_date.strftime('%B %d, %Y') if selected_view_date else 'N/A'}")
    
    # Use st.columns and call render_kpi_card within each column context
    kpi_cols_main_chw = st.columns(3)
    with kpi_cols_main_chw[0]:
        render_kpi_card("Visits Recorded", str(chw_kpis.get('visits_today',0)), "🚶‍♂️", 
                        help_text="Patient visits recorded on the selected date.")
    with kpi_cols_main_chw[1]:
        tasks_val = chw_kpis.get('tb_contacts_to_trace',0) + chw_kpis.get('sti_symptomatic_referrals',0)
        render_kpi_card("Key Disease Tasks", str(tasks_val), "📝", 
                        status="Moderate" if tasks_val > 2 else "Low", 
                        help_text="TB contacts to trace + STI symptomatic referrals.")
    with kpi_cols_main_chw[2]:
        risk_val = chw_kpis.get('avg_patient_risk_visited',0)
        render_kpi_card("Avg. Risk (Visited)", f"{risk_val:.0f}" if risk_val > 0 else "N/A", "🎯", 
                        status="High" if risk_val > 70 else "Moderate" if risk_val > 0 else "Low", 
                        help_text="Average AI risk score of patients visited today.")

    st.markdown("##### Patient Wellness Indicators (Visited Today)")
    kpi_cols_wellness_chw = st.columns(3)
    with kpi_cols_wellness_chw[0]:
        low_spo2_count = chw_kpis.get('patients_low_spo2_visited',0)
        render_kpi_card("Patients Low SpO2", str(low_spo2_count), "💨", 
                        status="High" if low_spo2_count > 0 else "Low", 
                        help_text=f"Patients visited with SpO2 < {app_config.SPO2_LOW_THRESHOLD_PCT}%.")
    with kpi_cols_wellness_chw[1]:
        fever_count = chw_kpis.get('patients_fever_visited',0)
        render_kpi_card("Patients w/ Fever", str(fever_count), "🔥", 
                        status="High" if fever_count > 0 else "Low", 
                        help_text=f"Patients visited with skin temp >= {app_config.SKIN_TEMP_FEVER_THRESHOLD_C}°C.")
    with kpi_cols_wellness_chw[2]:
        avg_steps_val = chw_kpis.get('avg_chw_steps',0)
        render_kpi_card("Avg. Patient Steps", f"{avg_steps_val:.0f}" if avg_steps_val > 0 else "N/A", "👣", 
                        status="Low" if avg_steps_val > 0 and avg_steps_val < 5000 else "Moderate", 
                        help_text="Average daily steps of patients in today's view (if data available).")
    st.markdown("---")

    tab_alerts_chw, tab_tasks_chw = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])
    # Ensure current_day_df is not empty before passing
    patient_alerts_df_chw = get_patient_alerts_for_chw(current_day_df, risk_threshold=app_config.RISK_THRESHOLDS['chw_alert_moderate']) if not current_day_df.empty else pd.DataFrame()

    with tab_alerts_chw:
        st.subheader("Critical Patient Alerts")
        if not patient_alerts_df_chw.empty:
            sorted_alerts_chw = patient_alerts_df_chw.sort_values(by='ai_risk_score', ascending=False)
            for _, alert_chw in sorted_alerts_chw.head(10).iterrows():
                status_alert = "High" if alert_chw.get('ai_risk_score', 0) >= app_config.RISK_THRESHOLDS['chw_alert_high'] else \
                               "Moderate" if alert_chw.get('ai_risk_score', 0) >= app_config.RISK_THRESHOLDS['chw_alert_moderate'] else "Low"
                render_traffic_light( # Direct call
                    f"Patient {alert_chw.get('patient_id','N/A')} ({alert_chw.get('condition','N/A')})", 
                    status_alert, 
                    details=f"Risk: {alert_chw.get('ai_risk_score', 0):.0f} | Reason: {alert_chw.get('alert_reason','N/A')}"
                )
        else: st.success("✅ No critical patient alerts for today based on current criteria.")
            
    with tab_tasks_chw:
        st.subheader("Prioritized Task List")
        if not patient_alerts_df_chw.empty:
            display_cols_chw = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today']
            actual_display_cols_chw = [col for col in display_cols_chw if col in patient_alerts_df_chw.columns]
            task_df_display_chw = patient_alerts_df_chw[actual_display_cols_chw].sort_values(by='ai_risk_score', ascending=False)
            st.dataframe(task_df_display_chw, use_container_width=True, height=350, 
                         column_config={"ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100)})
            csv_tasks_chw = task_df_display_chw.to_csv(index=False).encode('utf-8')
            st.download_button("Download Task List (CSV)", csv_tasks_chw, f"chw_tasks_{selected_view_date}.csv", "text/csv", key="chw_tasks_download_final")
        else: st.info("No specific tasks or follow-ups identified for today.")
            
    st.markdown("---")
    st.subheader(f"Overall Trends (Last {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} Days)")
    # Ensure health_df has data for trends and correct date filtering is applied
    trend_start_date = max_date_chw_page - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1)
    if trend_start_date < min_date_chw_page: trend_start_date = min_date_chw_page # clamp

    trend_df_chw = health_df[health_df['date'].dt.date >= trend_start_date].copy() if not health_df.empty else pd.DataFrame()

    if not trend_df_chw.empty:
        trend_cols_chw_page = st.columns(2)
        with trend_cols_chw_page[0]:
            risk_score_trend_chw = get_trend_data(trend_df_chw, 'ai_risk_score', period='D', agg_func='mean')
            if not risk_score_trend_chw.empty: st.plotly_chart(plot_annotated_line_chart(risk_score_trend_chw, "Daily Avg. Patient Risk Score", y_axis_title="Avg. Risk Score", height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
            else: st.caption("No risk score trend data.")
        with trend_cols_chw_page[1]:
            chw_visits_data = trend_df_chw[pd.to_numeric(trend_df_chw.get('chw_visit'), errors='coerce').fillna(0) == 1]
            visits_trend_chw = get_trend_data(chw_visits_data, 'chw_visit', period='D', agg_func='count')
            if not visits_trend_chw.empty: st.plotly_chart(plot_annotated_line_chart(visits_trend_chw, "Daily CHW Visits Recorded", y_axis_title="Number of Visits", height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
            else: st.caption("No CHW visits trend data.")
    else:
        st.info("Not enough historical data for overall trends.")
