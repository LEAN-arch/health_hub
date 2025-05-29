# pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary, # Updated to include wearable metrics
    get_patient_alerts_for_chw, # Updated for wearable alerts
    get_trend_data
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    render_traffic_light,
    plot_annotated_line_chart
)
import logging

st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

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
    min_date_chw = health_df['date'].min().date() if not health_df.empty and 'date' in health_df else pd.Timestamp('today').date() - pd.Timedelta(days=30)
    max_date_chw = health_df['date'].max().date() if not health_df.empty and 'date' in health_df else pd.Timestamp('today').date()
    
    selected_view_date = st.sidebar.date_input(
        "View Data For Date", max_date_chw, 
        min_value=min_date_chw, max_value=max_date_chw, 
        key="chw_view_date_final",
        help="Select date for daily summaries and tasks."
    )
    current_day_df = health_df[health_df['date'].dt.date == selected_view_date].copy() if not health_df.empty else pd.DataFrame()

    chw_kpis = get_chw_summary(current_day_df)
    
    st.subheader(f"Daily Snapshot: {selected_view_date.strftime('%B %d, %Y')}")
    kpi_cols_main = st.columns(3)
    with kpi_cols_main[0]: render_kpi_card("Visits Recorded", str(chw_kpis.get('visits_today',0)), "🚶‍♂️", help_text="Patient visits recorded today.")
    with kpi_cols_main[1]:
        tasks_val = chw_kpis.get('tb_contacts_to_trace',0) + chw_kpis.get('sti_symptomatic_referrals',0) # Example combined task
        render_kpi_card("Key Disease Tasks", str(tasks_val), "📝", status="Moderate" if tasks_val > 2 else "Low", help_text="TB contacts to trace + STI symptomatic referrals.")
    with kpi_cols_main[2]:
        risk_val = chw_kpis.get('avg_patient_risk_visited',0)
        render_kpi_card("Avg. Risk (Visited)", f"{risk_val:.0f}" if risk_val else "N/A", "🎯", status="High" if risk_val > 70 else "Moderate", help_text="Average AI risk score of patients visited today.")

    st.markdown("##### Patient Wellness Indicators (Visited Today)")
    kpi_cols_wellness = st.columns(3)
    with kpi_cols_wellness[0]:
        low_spo2_count = chw_kpis.get('patients_low_spo2_visited',0)
        render_kpi_card("Patients Low SpO2", str(low_spo2_count), "💨", status="High" if low_spo2_count > 0 else "Low", help_text=f"Patients visited with SpO2 < {app_config.SPO2_LOW_THRESHOLD_PCT}%.")
    with kpi_cols_wellness[1]:
        fever_count = chw_kpis.get('patients_fever_visited',0)
        render_kpi_card("Patients w/ Fever", str(fever_count), "🔥", status="High" if fever_count > 0 else "Low", help_text=f"Patients visited with skin temp >= {app_config.SKIN_TEMP_FEVER_THRESHOLD_C}°C.")
    with kpi_cols_wellness[2]:
        avg_steps_val = chw_kpis.get('avg_chw_steps',0) # This was in CHW summary, context might be patient or CHW's own
        render_kpi_card("Avg. Patient Steps", f"{avg_steps_val:.0f}", "👣", status="Low" if avg_steps_val < 5000 else "Moderate", help_text="Average daily steps of patients in today's view (if data available).")
    st.markdown("---")

    tab_alerts, tab_tasks = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])
    patient_alerts_df = get_patient_alerts_for_chw(current_day_df, risk_threshold=app_config.RISK_THRESHOLDS['chw_alert_moderate'])

    with tab_alerts:
        st.subheader("Critical Patient Alerts")
        if not patient_alerts_df.empty:
            # ... (Alert rendering logic using render_traffic_light as before) ...
            sorted_alerts = patient_alerts_df.sort_values(by='ai_risk_score', ascending=False);
            for _, alert in sorted_alerts.head(10).iterrows():
                status = "High" if alert['ai_risk_score'] >= app_config.RISK_THRESHOLDS['chw_alert_high'] else "Moderate" if alert['ai_risk_score'] >= app_config.RISK_THRESHOLDS['chw_alert_moderate'] else "Low"
                render_traffic_light(f"Patient {alert['patient_id']} ({alert['condition']})", status, details=f"Risk: {alert['ai_risk_score']:.0f} | Reason: {alert['alert_reason']}")
        else: st.success("✅ No critical patient alerts for today based on current criteria.")
            
    with tab_tasks:
        # ... (Task list display as before, ensure new alert columns are available if needed) ...
        st.subheader("Prioritized Task List");
        if not patient_alerts_df.empty:
            display_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today']
            actual_display_cols = [col for col in display_cols if col in patient_alerts_df.columns]
            task_df_display = patient_alerts_df[actual_display_cols].sort_values(by='ai_risk_score', ascending=False)
            st.dataframe(task_df_display, use_container_width=True, height=350, column_config={"ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100)})
            # ... (Download button) ...
        else: st.info("No specific tasks or follow-ups identified for today.")
            
    st.markdown("---")
    st.subheader("Overall Trends (Last 30 Days)") # Use a fixed window or filtered_health_records for trends
    trend_df_chw = health_df[health_df['date'] >= (health_df['date'].max() - pd.Timedelta(days=30))] if not health_df.empty else pd.DataFrame()

    if not trend_df_chw.empty:
        trend_cols_chw = st.columns(2)
        with trend_cols_chw[0]:
            # ... (Risk score trend as before) ...
            risk_score_trend_chw = get_trend_data(trend_df_chw, 'ai_risk_score', period='D', agg_func='mean')
            if not risk_score_trend_chw.empty: st.plotly_chart(plot_annotated_line_chart(risk_score_trend_chw, "Daily Avg. Patient Risk Score", y_axis_title="Avg. Risk Score"), use_container_width=True)
            else: st.caption("No risk score trend data.")
        with trend_cols_chw[1]:
            # ... (CHW visits trend as before) ...
            visits_trend_chw = get_trend_data(trend_df_chw[trend_df_chw['chw_visit']==1], 'chw_visit', period='D', agg_func='count')
            if not visits_trend_chw.empty: st.plotly_chart(plot_annotated_line_chart(visits_trend_chw, "Daily CHW Visits Recorded", y_axis_title="Number of Visits"), use_container_width=True)
            else: st.caption("No CHW visits trend data.")
    else:
        st.info("Not enough historical data for overall trends.")
