# pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_trend_data
)
from utils.ui_visualization_helpers import (
    get_kpi_card_html, # Changed from render_kpi_card
    get_traffic_light_html, # Changed from render_traffic_light
    render_html_row,    # New helper to render a row
    plot_annotated_line_chart
)
import logging # Import logging

# --- Page Configuration and Styling ---
st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css():
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: # pragma: no cover
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
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
    
    selected_view_date = st.sidebar.date_input(
        "View Data For Date", 
        max_date, 
        min_value=min_date, 
        max_value=max_date, 
        key="chw_view_date_filter_v2", # Ensure unique key
        help="Select the date for which you want to see daily summaries and tasks."
    )
    
    current_day_df = health_df[health_df['date'].dt.date == selected_view_date].copy()

    # --- KPIs ---
    chw_kpis = get_chw_summary(current_day_df)
    
    st.subheader(f"Daily Snapshot: {selected_view_date.strftime('%B %d, %Y')}")
    
    # Prepare HTML for each KPI card
    kpi_cards_html_list = []
    kpi_cards_html_list.append(
        get_kpi_card_html("Visits Recorded", str(chw_kpis['visits_today']), "🚶‍♂️", 
                          help_text="Number of patient visits recorded on the selected date.")
    )
    status_tasks = "Moderate" if chw_kpis['pending_tasks'] > 5 else "Low" if chw_kpis['pending_tasks'] > 0 else "Low"
    kpi_cards_html_list.append(
        get_kpi_card_html("Open Tasks", str(chw_kpis['pending_tasks']), "📝", status=status_tasks,
                          help_text="Total pending referrals and TB contact traces assigned.")
    )
    status_hr = "High" if chw_kpis['high_risk_followups'] > 3 else "Moderate" if chw_kpis['high_risk_followups'] > 0 else "Low"
    kpi_cards_html_list.append(
        get_kpi_card_html("High-Risk Follow-ups", str(chw_kpis['high_risk_followups']), "⚠️", status=status_hr,
                          help_text=f"Patients with risk score >= {app_config.RISK_THRESHOLDS['high']} needing attention.")
    )

    # Render the row of KPI cards
    render_html_row(kpi_cards_html_list)

    st.markdown("---")

    # --- Patient Alerts & Task List ---
    tab_alerts, tab_tasks = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])

    patient_alerts_df = get_patient_alerts_for_chw(current_day_df, risk_threshold=app_config.RISK_THRESHOLDS['chw_alert_moderate'])

    with tab_alerts:
        st.subheader("Critical Patient Alerts")
        if not patient_alerts_df.empty:
            alerts_html_list = []
            sorted_alerts = patient_alerts_df.sort_values(by='ai_risk_score', ascending=False)
            for _, alert in sorted_alerts.head(10).iterrows():
                status = "High" if alert['ai_risk_score'] >= app_config.RISK_THRESHOLDS['chw_alert_high'] else \
                         "Moderate" if alert['ai_risk_score'] >= app_config.RISK_THRESHOLDS['chw_alert_moderate'] else "Low"
                alerts_html_list.append(
                    get_traffic_light_html( # Use the HTML generator
                        f"Patient {alert['patient_id']} ({alert['condition']})", 
                        status, 
                        details=f"Risk: {alert['ai_risk_score']:.0f} | Reason: {alert['alert_reason']}"
                    )
                )
            # Render all traffic lights in one go if you want them stacked without `st.columns` issues
            # If just stacking them vertically, individual st.markdown calls for each is fine usually.
            # For this example, let's assume we stack them using individual calls as the issue was primarily with horizontal layouts.
            for html_item in alerts_html_list:
                st.markdown(html_item, unsafe_allow_html=True)
        else:
            st.success("✅ No critical patient alerts for the selected date based on current criteria.")
            
    with tab_tasks:
        st.subheader("Prioritized Task List")
        if not patient_alerts_df.empty:
            display_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status']
            actual_display_cols = [col for col in display_cols if col in patient_alerts_df.columns]
            
            task_df_display = patient_alerts_df[actual_display_cols].sort_values(by='ai_risk_score', ascending=False)
            st.dataframe(
                task_df_display, 
                use_container_width=True,
                height=350,
                column_config={
                    "ai_risk_score": st.column_config.ProgressColumn(
                        "AI Risk Score", help="Patient's current AI-calculated risk score.",
                        format="%d", min_value=0, max_value=100,
                    ),
                }
            )
            csv_tasks = task_df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Task List (CSV)", csv_tasks, f"chw_tasks_{selected_view_date}.csv", "text/csv", key="chw_tasks_download_v2")
        else:
            st.info("No specific tasks or follow-ups identified for the selected date.")
            
    st.markdown("---")

    # --- Trends ---
    st.subheader("Performance & Workload Trends (Overall)")
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        risk_score_trend = get_trend_data(health_df, 'ai_risk_score', period='W')
        if not risk_score_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(
                risk_score_trend, "Weekly Avg. Patient Risk Score Trend",
                y_axis_title="Avg. Risk Score",
                target_line=app_config.RISK_THRESHOLDS['moderate'], target_label=f"Moderate Threshold: {app_config.RISK_THRESHOLDS['moderate']}",
                height=app_config.DEFAULT_PLOT_HEIGHT
            ), use_container_width=True)
        else:
            st.caption("Not enough data for risk score trend.")

    with trend_cols[1]:
        chw_visits_df = health_df[health_df['chw_visit'] == 1].copy() # Use a copy for modifications
        # Ensure 'date' is datetime before grouping for trend
        if not chw_visits_df.empty and 'date' in chw_visits_df.columns:
            chw_visits_df['date'] = pd.to_datetime(chw_visits_df['date'], errors='coerce')
            chw_visits_df.dropna(subset=['date'], inplace=True)
        
            visits_trend = get_trend_data(chw_visits_df, 'chw_visit', agg_func='count', period='W')

            if not visits_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    visits_trend, "Weekly CHW Visits Recorded",
                    y_axis_title="Number of Visits",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else:
                st.caption("Not enough data for CHW visits trend.")
        else:
            st.caption("No CHW visit data available for trend.")
