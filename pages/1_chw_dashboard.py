# pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_trend_data
)
from utils.ui_visualization_helpers import (
    render_kpi_card, # Using the direct rendering function from helpers
    render_traffic_light,
    plot_annotated_line_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Get logger for this page

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600)
def get_chw_page_data():
    logger.info("Loading health records for CHW dashboard...")
    df = load_health_records()
    if df.empty: # pragma: no cover
        logger.error("Failed to load health records for CHW dashboard.")
    else:
        logger.info(f"Successfully loaded {len(df)} health records for CHW dashboard.")
    return df

health_df = get_chw_page_data()

# --- Main Page ---
if health_df.empty:
    st.error("🚨 Critical Error: Could not load health records. CHW Dashboard cannot be displayed. Please check data sources and configurations.")
    st.stop() # Stop execution if no data, as the dashboard relies on it.
else:
    st.title("🧑‍⚕️ Community Health Worker Dashboard")
    st.markdown("**Field Insights, Patient Prioritization & Wellness Monitoring**")
    st.markdown("---")

    # --- Robust Date Filter Setup ---
    st.sidebar.header("CHW Filters")
    min_date_chw_page = None
    max_date_chw_page = None
    default_selected_view_date = pd.Timestamp('today').date() # Sensible fallback

    if not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        # Ensure date column does not have all NaT values before calling min/max
        if health_df['date'].notna().any():
            min_date_dt_series = health_df['date'].min()
            max_date_dt_series = health_df['date'].max()

            if pd.notna(min_date_dt_series) and pd.notna(max_date_dt_series):
                min_date_chw_page = min_date_dt_series.date()
                max_date_chw_page = max_date_dt_series.date()
                default_selected_view_date = max_date_chw_page # Default to latest data date

                if min_date_chw_page > max_date_chw_page: # Should not happen if min/max logic is sound
                    min_date_chw_page = max_date_chw_page # pragma: no cover
            else: # All dates were NaT or some other issue
                logger.warning("Could not determine valid min/max dates from health_df['date']. Using default date range.")
                min_date_chw_page = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_VIEW * 10) # Wider fallback range
                max_date_chw_page = pd.Timestamp('today').date()
        else: # pragma: no cover
            logger.warning("All 'date' values in health_df are NaT. Using default date range for CHW filter.")
            min_date_chw_page = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_VIEW * 10)
            max_date_chw_page = pd.Timestamp('today').date()
    else: # pragma: no cover
        logger.warning("Health_df is empty or 'date' column is missing/invalid. Using default date range for CHW filter.")
        min_date_chw_page = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_VIEW * 10)
        max_date_chw_page = pd.Timestamp('today').date()


    selected_view_date = st.sidebar.date_input(
        "View Data For Date", 
        value=default_selected_view_date,
        min_value=min_date_chw_page, 
        max_value=max_date_chw_page, 
        key="chw_view_date_final_v3_complete", # Ensure unique key
        help="Select the date for which you want to see daily summaries and tasks."
    )
    
    # Filter data for the selected view date for KPIs and current task lists
    current_day_df = pd.DataFrame() # Initialize as empty
    if selected_view_date and not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        current_day_df = health_df[health_df['date'].dt.date == selected_view_date].copy()

    chw_kpis = get_chw_summary(current_day_df) # Handles empty current_day_df internally
    
    st.subheader(f"Daily Snapshot: {selected_view_date.strftime('%B %d, %Y') if selected_view_date else 'N/A'}")
    
    kpi_cols_main_chw = st.columns(3)
    with kpi_cols_main_chw[0]:
        render_kpi_card("Visits Recorded", str(chw_kpis.get('visits_today',0)), "🚶‍♂️", 
                        help_text="Patient visits recorded on the selected date.")
    with kpi_cols_main_chw[1]:
        tasks_val = chw_kpis.get('tb_contacts_to_trace',0) + chw_kpis.get('sti_symptomatic_referrals',0)
        render_kpi_card("Key Disease Tasks", str(tasks_val), "📝", 
                        status="Moderate" if tasks_val > 2 else "Low" if tasks_val > 0 else "Low", 
                        help_text="TB contacts to trace + STI symptomatic referrals.")
    with kpi_cols_main_chw[2]:
        risk_val_visited = chw_kpis.get('avg_patient_risk_visited',0.0)
        # Determine if N/A should be shown for average risk
        patients_visited_count = current_day_df[current_day_df.get('chw_visit', pd.Series(dtype=int)) == 1].shape[0] if not current_day_df.empty else 0
        risk_display_val = f"{risk_val_visited:.0f}" if patients_visited_count > 0 and pd.notna(risk_val_visited) else "N/A"
        risk_status = "High" if risk_val_visited > 70 else "Moderate" if risk_val_visited > 0 else "Low"
        if risk_display_val == "N/A": risk_status="Low" # Or some other default if N/A
        
        render_kpi_card("Avg. Risk (Visited)", risk_display_val, "🎯", 
                        status=risk_status, 
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
        avg_steps_val_kpi = chw_kpis.get('avg_chw_steps',0.0)
        patients_with_steps_data = current_day_df[current_day_df.get('chw_visit', pd.Series(dtype=int)) == 1 & current_day_df.get('avg_daily_steps', pd.Series(dtype=float)).notna()].shape[0] if not current_day_df.empty else 0
        steps_display_val = f"{avg_steps_val_kpi:.0f}" if patients_with_steps_data > 0 and pd.notna(avg_steps_val_kpi) else "N/A"
        steps_status = "Low" if avg_steps_val_kpi > 0 and avg_steps_val_kpi < (app_config.TARGET_DAILY_STEPS * 0.7) else "Moderate"
        if steps_display_val == "N/A": steps_status = "Low"

        render_kpi_card("Avg. Patient Steps", steps_display_val, "👣", 
                        status=steps_status, 
                        help_text=f"Avg. daily steps of visited patients with data. Target: {app_config.TARGET_DAILY_STEPS} steps.")
    st.markdown("---")

    tab_alerts_chw, tab_tasks_chw = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])
    
    patient_alerts_df_chw = pd.DataFrame() # Initialize
    if not current_day_df.empty:
        patient_alerts_df_chw = get_patient_alerts_for_chw(current_day_df, risk_threshold=app_config.RISK_THRESHOLDS['chw_alert_moderate'])

    with tab_alerts_chw:
        st.subheader("Critical Patient Alerts")
        if not patient_alerts_df_chw.empty:
            sorted_alerts_chw = patient_alerts_df_chw.sort_values(by='ai_risk_score', ascending=False)
            for _, alert_chw in sorted_alerts_chw.head(10).iterrows(): # Show top 10 alerts
                ai_risk_score_alert = alert_chw.get('ai_risk_score', 0) # Default to 0 if missing
                status_alert = "High" if ai_risk_score_alert >= app_config.RISK_THRESHOLDS['chw_alert_high'] else \
                               "Moderate" if ai_risk_score_alert >= app_config.RISK_THRESHOLDS['chw_alert_moderate'] else "Low"
                render_traffic_light(
                    f"Patient {alert_chw.get('patient_id','N/A')} ({alert_chw.get('condition','N/A')})", 
                    status_alert, 
                    details=f"Risk: {ai_risk_score_alert:.0f} | Reason: {alert_chw.get('alert_reason','N/A')}"
                )
        else: st.success("✅ No critical patient alerts for today based on current criteria.")
            
    with tab_tasks_chw:
        st.subheader("Prioritized Task List")
        if not patient_alerts_df_chw.empty:
            display_cols_chw = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today']
            actual_display_cols_chw = [col for col in display_cols_chw if col in patient_alerts_df_chw.columns]
            
            task_df_display_chw = patient_alerts_df_chw[actual_display_cols_chw].sort_values(by='ai_risk_score', ascending=False)
            st.dataframe(
                task_df_display_chw, 
                use_container_width=True,
                height=350, # Adjust height as needed
                column_config={
                    "ai_risk_score": st.column_config.ProgressColumn(
                        "AI Risk Score", 
                        help="Patient's AI-calculated risk score.",
                        format="%d", min_value=0, max_value=100,
                    ),
                    "min_spo2_pct": st.column_config.NumberColumn("Min SpO2 (%)", format="%d%%"),
                    "max_skin_temp_celsius": st.column_config.NumberColumn("Max Temp (°C)", format="%.1f°C"),
                    "fall_detected_today": st.column_config.NumberColumn("Falls Today", format="%d"),
                }
            )
            # Download button
            try:
                csv_tasks_chw = task_df_display_chw.to_csv(index=False).encode('utf-8')
                safe_date_str_chw = selected_view_date.strftime('%Y%m%d') if selected_view_date else "all_dates"
                st.download_button("Download Task List (CSV)", csv_tasks_chw, f"chw_tasks_{safe_date_str_chw}.csv", "text/csv", key="chw_tasks_download_final_v2") # Unique key
            except Exception as e_csv: # pragma: no cover
                logger.error(f"Error preparing CHW task list for download: {e_csv}")
                st.warning("Could not prepare task list for download.")
        else: st.info("No specific tasks or follow-ups identified for today.")
            
    st.markdown("---")
    st.subheader(f"Overall Trends (Last {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} Days)")
    trend_df_chw = pd.DataFrame() # Initialize

    if selected_view_date and min_date_chw_page and not health_df.empty: # Ensure selected_view_date (as max for trend window) is valid
        trend_lookback_date = selected_view_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
        trend_start_date_clamped = max(trend_lookback_date, min_date_chw_page) # Clamp to data min_date

        trend_df_chw = health_df[
            (health_df['date'].dt.date >= trend_start_date_clamped) &
            (health_df['date'].dt.date <= selected_view_date) # Trend up to selected view date
        ].copy()
    
    if not trend_df_chw.empty:
        trend_cols_chw_page = st.columns(2)
        with trend_cols_chw_page[0]:
            risk_score_trend_chw = get_trend_data(trend_df_chw, 'ai_risk_score', period='D', agg_func='mean')
            if not risk_score_trend_chw.empty: 
                st.plotly_chart(plot_annotated_line_chart(
                    risk_score_trend_chw, "Daily Avg. Patient Risk Score (Trend Window)", 
                    y_axis_title="Avg. Risk Score", height=app_config.DEFAULT_PLOT_HEIGHT-20, show_anomalies=False # Fewer anomalies for cleaner trend
                ), use_container_width=True)
            else: st.caption("No risk score trend data for the period.")
        with trend_cols_chw_page[1]:
            # Ensure chw_visit is numeric before filtering
            trend_df_chw['chw_visit_numeric'] = pd.to_numeric(trend_df_chw.get('chw_visit'), errors='coerce').fillna(0)
            chw_visits_data = trend_df_chw[trend_df_chw['chw_visit_numeric'] == 1]
            visits_trend_chw = get_trend_data(chw_visits_data, 'chw_visit_numeric', period='D', agg_func='count')
            if not visits_trend_chw.empty: 
                st.plotly_chart(plot_annotated_line_chart(
                    visits_trend_chw, "Daily CHW Visits Recorded (Trend Window)", 
                    y_axis_title="Number of Visits", height=app_config.DEFAULT_PLOT_HEIGHT-20, show_anomalies=False
                ), use_container_width=True)
            else: st.caption("No CHW visits trend data for the period.")
    else:
        st.info(f"Not enough historical data (min {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} days needed) ending on {selected_view_date.strftime('%Y-%m-%d') if selected_view_date else 'N/A'} for overall trends.")
