# pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw, # New more specific function
    get_trend_data
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    render_traffic_light,
    plot_annotated_line_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")

def load_css():
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600)
def get_chw_page_data():
    health_df = load_health_records()
    return health_df

health_df = get_chw_page_data()

# --- Main Page ---
if health_df.empty:
    st.error("🚨 Critical Error: Could not load health records. CHW Dashboard cannot be displayed. Please check data sources and configurations.")
else:
    st.title("🧑‍⚕️ Community Health Worker Dashboard")
    st.markdown("**Actionable Field Insights & Patient Management**")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("CHW Filters")
    min_date = health_df['date'].min().date()
    max_date = health_df['date'].max().date()
    
    # For point-in-time KPIs, CHWs usually look at "today" or a recent single day.
    selected_view_date = st.sidebar.date_input(
        "View Data For Date", 
        max_date, # Default to the latest date in the dataset
        min_value=min_date, 
        max_value=max_date, 
        key="chw_view_date_filter",
        help="Select the date for which you want to see daily summaries and tasks."
    )
    
    # Filter data for the selected view date for KPIs and current task lists
    current_day_df = health_df[health_df['date'].dt.date == selected_view_date].copy()

    # --- KPIs ---
    chw_kpis = get_chw_summary(current_day_df) # Pass the filtered df for the selected day
    
    st.subheader(f"Daily Snapshot: {selected_view_date.strftime('%B %d, %Y')}")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        render_kpi_card("Visits Recorded", str(chw_kpis['visits_today']), "🚶‍♂️", 
                        help_text="Number of patient visits recorded on the selected date.")
    with kpi_cols[1]:
        status_tasks = "Moderate" if chw_kpis['pending_tasks'] > 5 else "Low" if chw_kpis['pending_tasks'] > 0 else "Low" # Low if 0 too
        render_kpi_card("Open Tasks", str(chw_kpis['pending_tasks']), "📝", status=status_tasks,
                        help_text="Total pending referrals and TB contact traces assigned.")
    with kpi_cols[2]:
        status_hr = "High" if chw_kpis['high_risk_followups'] > 3 else "Moderate" if chw_kpis['high_risk_followups'] > 0 else "Low"
        render_kpi_card("High-Risk Follow-ups", str(chw_kpis['high_risk_followups']), "⚠️", status=status_hr,
                        help_text=f"Patients with risk score >= {app_config.RISK_THRESHOLDS['high']} needing attention.")

    st.markdown("---")

    # --- Patient Alerts & Task List ---
    tab_alerts, tab_tasks = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])

    # Use get_patient_alerts_for_chw for the current view date context
    patient_alerts_df = get_patient_alerts_for_chw(current_day_df, risk_threshold=app_config.RISK_THRESHOLDS['chw_alert_moderate'])

    with tab_alerts:
        st.subheader("Critical Patient Alerts")
        if not patient_alerts_df.empty:
            # Sort by risk score to show most critical first
            sorted_alerts = patient_alerts_df.sort_values(by='ai_risk_score', ascending=False)
            for _, alert in sorted_alerts.head(10).iterrows(): # Show top 10 alerts
                status = "High" if alert['ai_risk_score'] >= app_config.RISK_THRESHOLDS['chw_alert_high'] else \
                         "Moderate" if alert['ai_risk_score'] >= app_config.RISK_THRESHOLDS['chw_alert_moderate'] else "Low"
                render_traffic_light(
                    f"Patient {alert['patient_id']} ({alert['condition']})", 
                    status, 
                    details=f"Risk: {alert['ai_risk_score']:.0f} | Reason: {alert['alert_reason']}"
                )
        else:
            st.success("✅ No critical patient alerts for the selected date based on current criteria.")
            
    with tab_tasks:
        st.subheader("Prioritized Task List")
        if not patient_alerts_df.empty:
            display_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status']
            # Ensure all display_cols exist in patient_alerts_df
            actual_display_cols = [col for col in display_cols if col in patient_alerts_df.columns]
            
            task_df_display = patient_alerts_df[actual_display_cols].sort_values(by='ai_risk_score', ascending=False)
            st.dataframe(
                task_df_display, 
                use_container_width=True,
                height=350,
                column_config={ # Example of customizing column display
                    "ai_risk_score": st.column_config.ProgressColumn(
                        "AI Risk Score",
                        help="Patient's current AI-calculated risk score.",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )
            csv_tasks = task_df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Task List (CSV)", csv_tasks, f"chw_tasks_{selected_view_date}.csv", "text/csv", key="chw_tasks_download")
        else:
            st.info("No specific tasks or follow-ups identified for the selected date.")
            
    st.markdown("---")

    # --- Trends (using the full historical health_df) ---
    st.subheader("Performance & Workload Trends (Overall)")
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        # Overall risk score trend of patients encountered by CHWs (or all patients in their zones)
        # For simplicity, using the mean risk score of all patients over time.
        risk_score_trend = get_trend_data(health_df, 'ai_risk_score', period='W') # Weekly trend
        if not risk_score_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(
                risk_score_trend,
                "Weekly Avg. Patient Risk Score Trend",
                y_axis_title="Avg. Risk Score",
                target_line=app_config.RISK_THRESHOLDS['moderate'], target_label=f"Moderate Threshold: {app_config.RISK_THRESHOLDS['moderate']}",
                height=app_config.DEFAULT_PLOT_HEIGHT
            ), use_container_width=True)
        else:
            st.caption("Not enough data for risk score trend.")

    with trend_cols[1]:
        # Trend of CHW visits
        chw_visits_df = health_df[health_df['chw_visit'] == 1]
        visits_trend = get_trend_data(chw_visits_df, 'chw_visit', agg_func='count', period='W') # Count of visits per week

        if not visits_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(
                visits_trend,
                "Weekly CHW Visits Recorded",
                y_axis_title="Number of Visits",
                height=app_config.DEFAULT_PLOT_HEIGHT
            ), use_container_width=True)
        else:
            st.caption("Not enough data for CHW visits trend.")
