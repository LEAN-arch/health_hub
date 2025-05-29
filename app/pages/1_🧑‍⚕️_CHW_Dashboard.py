import streamlit as st
import pandas as pd
from utils.data_processor import load_health_data, get_chw_summary, get_patient_alerts, get_trend_data
from utils.viz_helper import render_kpi_card, render_traffic_light, plot_annotated_line_chart
import os

# Page configuration
st.set_page_config(page_title="CHW Dashboard", layout="wide")

# Function to load CSS
def load_css(file_name):
    # Correct path relative to this file's location if style.css is in the root
    # Assuming this file is in health_hub/pages/ and style.css is in health_hub/
    abs_path = os.path.join(os.path.dirname(__file__), "..", file_name)
    if os.path.exists(abs_path):
        with open(abs_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CHW CSS file not found: {abs_path}")
load_css("style.css")


# Load Data
health_df = load_health_data()

if health_df.empty:
    st.error("Could not load health data. CHW Dashboard cannot be displayed.")
else:
    st.title("🧑‍⚕️ Community Health Worker Dashboard")
    st.markdown("**Actionable Field Insights & Patient Management**")
    st.markdown("---")

    # --- Sidebar Filters (Example for CHW - could be patient status, risk level) ---
    st.sidebar.header("CHW Filters")
    # For CHW, might filter by specific tasks or patient segments in a real app
    # For now, we use all data for summaries
    
    min_date = health_df['date'].min().date()
    max_date = health_df['date'].max().date()
    
    selected_date_chw = st.sidebar.date_input(
        "View Data Up To", 
        max_date, 
        min_value=min_date, 
        max_value=max_date, 
        key="chw_date_filter"
    )
    
    # Filter data based on selected date (for current view, not trends over time)
    current_view_df = health_df[health_df['date'] <= pd.to_datetime(selected_date_chw)]


    # --- KPIs ---
    chw_kpis = get_chw_summary(current_view_df) # Use current_view_df for point-in-time KPIs
    
    st.subheader("Daily Snapshot & Key Tasks")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        render_kpi_card("Visits Today", str(chw_kpis['visits_today']), "🚶‍♂️", help_text="Number of patient visits recorded today.")
    with kpi_cols[1]:
        render_kpi_card("Pending Tasks", str(chw_kpis['pending_tasks']), "📝", status="Moderate" if chw_kpis['pending_tasks'] > 5 else "Low", help_text="Sum of pending referrals and TB contact traces.")
    with kpi_cols[2]:
        render_kpi_card("High-Risk Follow-ups", str(chw_kpis['high_risk_followups']), "⚠️", status="High" if chw_kpis['high_risk_followups'] > 3 else "Moderate", help_text="Patients with high risk scores needing attention.")

    st.markdown("---")

    # --- Patient Alerts & Task List ---
    # Use tabs for better organization
    tab1, tab2 = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])

    with tab1:
        st.subheader("Critical Patient Alerts")
        patient_alerts_df = get_patient_alerts(current_view_df, risk_threshold=70) # Use current_view_df
        if not patient_alerts_df.empty:
            for _, alert in patient_alerts_df.head(5).iterrows(): # Show top 5
                status = "High" if alert['ai_risk_score'] >= 80 else "Moderate" if alert['ai_risk_score'] >=60 else "Low"
                render_traffic_light(
                    f"Patient {alert['patient_id']} ({alert['condition']})", 
                    status, 
                    details=f"Risk: {alert['ai_risk_score']:.0f} | {alert['alert_reason']}"
                )
        else:
            st.info("No critical patient alerts at this time.")
            
    with tab2:
        st.subheader("Detailed Task List / Patient Follow-ups")
        # More comprehensive list from alerts or other task generation logic
        if not patient_alerts_df.empty:
            display_cols = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason']
            st.dataframe(
                patient_alerts_df[display_cols].sort_values(by='ai_risk_score', ascending=False), 
                use_container_width=True,
                height=300
            )
            # Add download for tasks
            csv_tasks = patient_alerts_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button("Download Task List (CSV)", csv_tasks, "chw_task_list.csv", "text/csv", key="chw_tasks_download")
        else:
            st.info("No tasks or follow-ups identified based on current filters.")
            
    st.markdown("---")

    # --- Trends ---
    st.subheader("Performance & Workload Trends")
    trend_cols = st.columns(2)
    
    # Use original health_df for trends over full history
    with trend_cols[0]:
        risk_score_trend = get_trend_data(health_df, 'ai_risk_score', period='D')
        if not risk_score_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(
                risk_score_trend,
                "Avg. Patient Risk Score Trend",
                y_axis_title="Avg. Risk Score",
                target_line=70, target_label="High Risk Threshold"
            ), use_container_width=True)
        else:
            st.caption("Not enough data for risk score trend.")

    with trend_cols[1]:
        # Create a 'task_completed' column for trend (mocked logic)
        health_df_copy = health_df.copy() # Work on a copy for trends
        health_df_copy['task_metric'] = health_df_copy['referral_status'].apply(lambda x: 1 if x == 'Completed' else 0) \
                                     + health_df_copy['tb_contact_traced'] # Example metric for tasks
        
        tasks_trend = get_trend_data(health_df_copy, 'task_metric', period='D')
        if not tasks_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(
                tasks_trend,
                "Daily Tasks Completed/Followed-up", # Adjusted title
                y_axis_title="Tasks Metric"
            ), use_container_width=True)
        else:
            st.caption("Not enough data for tasks trend.")
