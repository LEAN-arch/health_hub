# health_hub/pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from config import app_config # Centralized configuration
from utils.core_data_processing import (
    load_health_records, # Robust data loading
    get_chw_summary,      # Calculates CHW-specific KPIs for a given day
    get_patient_alerts_for_chw, # Generates prioritized alerts/tasks for CHW
    get_trend_data        # Generic function for time-series trend data
)
from utils.ui_visualization_helpers import (
    render_kpi_card,      # Renders styled KPI cards
    render_traffic_light, # Renders traffic light style alerts
    plot_annotated_line_chart # For plotting trends
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Logger for this page

@st.cache_resource # Cache CSS loading
def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("CHW Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"CHW Dashboard: CSS file not found at {app_config.STYLE_CSS}. Default Streamlit styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS) # Cache data loading
def get_chw_dashboard_data():
    logger.info("CHW Dashboard: Attempting to load health records...")
    health_df = load_health_records() # Uses the robust loader from core_data_processing
    if health_df.empty:
        logger.error("CHW Dashboard: Failed to load health records or returned an empty DataFrame.")
    else:
        logger.info(f"CHW Dashboard: Successfully loaded {len(health_df)} health records.")
    return health_df

health_df_chw_main = get_chw_dashboard_data()

# --- Main Page Rendering ---
if health_df_chw_main is None or health_df_chw_main.empty:
    st.error("🚨 **Critical Error:** Could not load necessary health records for the CHW Dashboard. Please verify data sources and configurations in `app_config.py`.")
    logger.critical("CHW Dashboard cannot render: health_df_chw_main is None or empty.")
    st.stop() # Halt execution if essential data is missing
else:
    st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
    st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**")
    st.markdown("---") # Visual separator

    # --- Sidebar Filters ---
    st.sidebar.header("🗓️ CHW Filters") # Emoji for visual appeal
    # Determine available date range from the data
    if 'date' in health_df_chw_main.columns and health_df_chw_main['date'].notna().any():
        min_date_available = health_df_chw_main['date'].min().date()
        max_date_available = health_df_chw_main['date'].max().date()
        default_selected_date = max_date_available # Default to the most recent data date
    else: # Fallback if date column is problematic
        logger.warning("CHW Dashboard: 'date' column issues in health_df. Using default date range for filter.")
        min_date_available = pd.Timestamp('today').date() - pd.Timedelta(days=90)
        max_date_available = pd.Timestamp('today').date()
        default_selected_date = max_date_available
    
    if min_date_available > max_date_available: # Safety check
        min_date_available = max_date_available


    selected_view_date = st.sidebar.date_input(
        "View Data For Date:",
        value=default_selected_date,
        min_value=min_date_available,
        max_value=max_date_available,
        key="chw_daily_view_date_selector", # Unique key
        help="Select the date for which you want to view daily summaries, tasks, and patient alerts."
    )

    # Filter DataFrame for the selected date to get `current_day_chw_df`
    current_day_chw_df = pd.DataFrame() # Initialize as empty
    if selected_view_date and 'date' in health_df_chw_main.columns:
        # Ensure 'date' column is in date objects for comparison, if not already
        health_df_chw_main['date_obj'] = pd.to_datetime(health_df_chw_main['date']).dt.date
        current_day_chw_df = health_df_chw_main[health_df_chw_main['date_obj'] == selected_view_date].copy()

    # Get CHW summary KPIs for the selected day
    chw_daily_kpis = get_chw_summary(current_day_chw_df) # This function handles empty df

    # --- Daily Snapshot KPIs ---
    st.subheader(f"Daily Snapshot: {selected_view_date.strftime('%A, %B %d, %Y') if selected_view_date else 'N/A'}")
    
    kpi_cols_chw_overview = st.columns(4) # Four KPIs for main overview
    with kpi_cols_chw_overview[0]:
        visits_val = chw_daily_kpis.get('visits_today', 0)
        render_kpi_card("Visits Today", str(visits_val), "🚶‍♀️", # Icon representing walking/visits
                        status="Low" if visits_val < 5 else ("Moderate" if visits_val < 10 else "High"), # Status based on visit count
                        help_text="Total patient visits recorded by the CHW on the selected date.")
    with kpi_cols_chw_overview[1]:
        key_tasks = chw_daily_kpis.get('tb_contacts_to_trace_today', 0) + chw_daily_kpis.get('sti_symptomatic_referrals_today', 0)
        render_kpi_card("Key Disease Tasks", str(key_tasks), "📋", # Icon for tasks/clipboard
                        status="High" if key_tasks > 5 else ("Moderate" if key_tasks > 0 else "Low"),
                        help_text="Sum of TB contacts needing tracing and STI symptomatic patients needing referral today.")
    with kpi_cols_chw_overview[2]:
        avg_risk_visited = chw_daily_kpis.get('avg_patient_risk_visited_today', 0.0)
        risk_display_text = f"{avg_risk_visited:.0f}" if pd.notna(avg_risk_visited) and avg_risk_visited > 0 else "N/A"
        risk_semantic = "High" if avg_risk_visited >= app_config.RISK_THRESHOLDS['high'] else \
                        "Moderate" if avg_risk_visited >= app_config.RISK_THRESHOLDS['moderate'] else "Low"
        if risk_display_text == "N/A": risk_semantic = "Neutral"
        render_kpi_card("Avg. Risk (Visited)", risk_display_text, "🎯", status=risk_semantic,
                        help_text="Average AI-calculated risk score of unique patients visited today.")
    with kpi_cols_chw_overview[3]:
        followups_needed = chw_daily_kpis.get('high_risk_followups_today', 0)
        render_kpi_card("High-Risk Follow-ups", str(followups_needed), "⚠️",
                        status="High" if followups_needed > 2 else ("Moderate" if followups_needed > 0 else "Low"),
                        help_text="Number of unique high-risk patients (from today's records) requiring CHW attention.")

    # --- Patient Wellness Indicators (for patients visited today) ---
    st.markdown("##### Patient Wellness Indicators (For Visited Patients)")
    kpi_cols_chw_wellness = st.columns(3)
    with kpi_cols_chw_wellness[0]:
        low_spo2 = chw_daily_kpis.get('patients_low_spo2_visited_today', 0)
        render_kpi_card("Low SpO2 Alerts", str(low_spo2), "💨", # Breathing/oxygen icon
                        status="High" if low_spo2 > 0 else "Low", # Any patient with low SpO2 is a "High" alert status for this KPI
                        help_text=f"Number of visited patients with SpO2 < {app_config.SPO2_LOW_THRESHOLD_PCT}%.")
    with kpi_cols_chw_wellness[1]:
        fever = chw_daily_kpis.get('patients_fever_visited_today', 0)
        render_kpi_card("Fever Alerts", str(fever), "🔥", status="High" if fever > 0 else "Low",
                        help_text=f"Number of visited patients with skin temperature ≥ {app_config.SKIN_TEMP_FEVER_THRESHOLD_C}°C.")
    with kpi_cols_chw_wellness[2]:
        avg_steps_val = chw_daily_kpis.get('avg_patient_steps_visited_today', 0.0)
        steps_display_text = f"{avg_steps_val:,.0f}" if pd.notna(avg_steps_val) and avg_steps_val > 0 else "N/A"
        steps_status = "Low" if avg_steps_val < (app_config.TARGET_DAILY_STEPS * 0.6) else \
                       "Moderate" if avg_steps_val < app_config.TARGET_DAILY_STEPS else "High" # Good if target met
        if steps_display_text == "N/A": steps_status = "Neutral"
        render_kpi_card("Avg. Patient Steps", steps_display_text, "👣", status=steps_status,
                        help_text=f"Average daily steps of visited patients. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    st.markdown("---") # Visual separator

    # --- Tabs for Alerts and Task List ---
    tab_alerts, tab_tasks = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])

    # Generate patient alerts for the CHW for the selected day
    # The get_patient_alerts_for_chw function should prioritize and provide reasons.
    patient_alerts_tasks_df = get_patient_alerts_for_chw(
        current_day_chw_df,
        risk_threshold_moderate=app_config.RISK_THRESHOLDS['chw_alert_moderate']
    )

    with tab_alerts:
        st.subheader("Critical Patient Alerts for Today")
        if not patient_alerts_tasks_df.empty:
            # Display top N alerts using the traffic light component for quick visual assessment
            for _, alert_row in patient_alerts_tasks_df.head(15).iterrows(): # Show top 15 most urgent alerts
                risk_score = alert_row.get('ai_risk_score', 0)
                alert_status_for_light = "High" if risk_score >= app_config.RISK_THRESHOLDS['chw_alert_high'] else \
                                         "Moderate" if risk_score >= app_config.RISK_THRESHOLDS['chw_alert_moderate'] else "Low"
                
                # Build detailed string for the traffic light
                alert_details_parts = []
                if pd.notna(risk_score) : alert_details_parts.append(f"Risk: {risk_score:.0f}")
                if pd.notna(alert_row.get('min_spo2_pct')): alert_details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
                if pd.notna(alert_row.get('max_skin_temp_celsius')): alert_details_parts.append(f"Temp: {alert_row['max_skin_temp_celsius']:.1f}°C")
                if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0:
                    alert_details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")

                alert_message = f"Patient {alert_row.get('patient_id','N/A')} - {alert_row.get('condition','N/A')} needs attention."
                alert_detail_string = alert_row.get('alert_reason', 'Review Case') + (" | " + " / ".join(alert_details_parts) if alert_details_parts else "")

                render_traffic_light(
                    message=alert_message,
                    status=alert_status_for_light,
                    details=alert_detail_string
                )
        else:
            st.success("✅ No critical patient alerts identified for today based on current criteria.")

    with tab_tasks:
        st.subheader("Prioritized Task List for Today")
        if not patient_alerts_tasks_df.empty:
            # Display a more detailed table for the task list
            task_list_cols_to_show = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason',
                                      'referral_status', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today', 'priority_score']
            task_df_display = patient_alerts_tasks_df[[col for col in task_list_cols_to_show if col in patient_alerts_tasks_df.columns]].copy()

            st.dataframe(
                task_df_display,
                use_container_width=True,
                height=450, # Adjust height as needed
                column_config={
                    "patient_id": st.column_config.TextColumn("Patient ID", help="Unique Patient Identifier"),
                    "zone_id": st.column_config.TextColumn("Zone"),
                    "ai_risk_score": st.column_config.ProgressColumn(
                        "AI Risk", help="Patient's AI-calculated risk score (0-100).",
                        format="%d", min_value=0, max_value=100, width="medium"
                    ),
                    "alert_reason": st.column_config.TextColumn("Primary Alert / Task Reason", width="large"),
                    "min_spo2_pct": st.column_config.NumberColumn("Min SpO2 (%)", format="%d%%", help="Lowest SpO2 reading."),
                    "max_skin_temp_celsius": st.column_config.NumberColumn("Max Temp (°C)", format="%.1f°C", help="Highest skin temperature."),
                    "fall_detected_today": st.column_config.NumberColumn("Falls Today", format="%d", help="Number of falls detected."),
                    "priority_score": st.column_config.NumberColumn("Priority", help="Calculated task priority score (higher is more urgent).", format="%d")
                },
                hide_index=True # Cleaner look without the pandas index
            )
            # Download button for the task list
            try:
                csv_chw_tasks = task_df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Task List (CSV)",
                    data=csv_chw_tasks,
                    file_name=f"chw_tasks_{selected_view_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="chw_task_list_download_button" # Unique key
                )
            except Exception as e_csv_download_chw: # pragma: no cover
                logger.error(f"CHW Dashboard: Error preparing task list for CSV download: {e_csv_download_chw}")
                st.warning("Could not prepare the task list for download at this time.")
        else:
            st.info("No specific tasks or follow-ups identified from the alerts list for today.")

    # --- Trend Charts ---
    st.markdown("---") # Visual separator
    st.subheader(f"Overall Patient Wellness & Activity Trends (Last {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} Days ending {selected_view_date.strftime('%B %d, %Y')})")

    # Filter main DataFrame for the trend period ending on the selected_view_date
    trend_period_end_date = selected_view_date
    trend_period_start_date = trend_period_end_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
    
    # Ensure 'date_obj' exists (created earlier for current_day_chw_df)
    if 'date_obj' not in health_df_chw_main.columns and 'date' in health_df_chw_main.columns:
        health_df_chw_main['date_obj'] = pd.to_datetime(health_df_chw_main['date']).dt.date
        
    chw_trend_df_filtered = health_df_chw_main[
        (health_df_chw_main['date_obj'] >= trend_period_start_date) &
        (health_df_chw_main['date_obj'] <= trend_period_end_date)
    ].copy() if 'date_obj' in health_df_chw_main else pd.DataFrame()


    if not chw_trend_df_filtered.empty:
        trend_cols_chw = st.columns(2)
        with trend_cols_chw[0]:
            # Trend of Average Patient Risk Score (for all patients in data, not just visited)
            overall_risk_trend = get_trend_data(chw_trend_df_filtered, 'ai_risk_score', date_col='date', period='D', agg_func='mean')
            if not overall_risk_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    overall_risk_trend, "Daily Avg. Patient Risk Score (All Patients)",
                    y_axis_title="Avg. Risk Score", height=app_config.COMPACT_PLOT_HEIGHT,
                    target_line=app_config.TARGET_PATIENT_RISK_SCORE, target_label="Target Avg. Risk",
                    show_anomalies=True, date_format="%d %b"
                ), use_container_width=True)
            else: st.caption("No overall risk score trend data available for this period.")

        with trend_cols_chw[1]:
            # Trend of Daily CHW Visits (unique patients visited)
            # Need chw_visit column; get_trend_data uses 'patient_id' with nunique for this
            chw_visits_trend_data = get_trend_data(
                chw_trend_df_filtered[pd.to_numeric(chw_trend_df_filtered.get('chw_visit'), errors='coerce').fillna(0) == 1], # Filter for records where CHW visit occurred
                value_col='patient_id', date_col='date', period='D', agg_func='nunique'
            )
            if not chw_visits_trend_data.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    chw_visits_trend_data, "Daily Unique Patients Visited by CHW",
                    y_axis_title="Number of Patients Visited", height=app_config.COMPACT_PLOT_HEIGHT,
                    show_anomalies=True, date_format="%d %b"
                ), use_container_width=True)
            else: st.caption("No CHW visits trend data available for this period (ensure 'chw_visit' column has data).")
    else:
        st.info(f"Not enough historical data (min {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} days needed) ending on {selected_view_date.strftime('%Y-%m-%d')} for overall trends display.")
