# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_clinic_summary,
    get_trend_data,
    get_supply_forecast_data, # Assuming this is enhanced
    get_patient_alerts_for_clinic # New more specific function
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_donut_chart,
    plot_annotated_line_chart,
    plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")

def load_css():
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600)
def get_clinic_page_data():
    health_df = load_health_records()
    return health_df

health_df = get_clinic_page_data()

# --- Main Page ---
if health_df.empty:
    st.error("🚨 Critical Error: Could not load health records. Clinic Dashboard cannot be displayed.")
else:
    st.title("🏥 Clinic Operations Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, and Resource Management**")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("Clinic Filters")
    min_date = health_df['date'].min().date()
    max_date = health_df['date'].max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [max_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1), max_date],
        min_value=min_date,
        max_value=max_date,
        key="clinic_date_range_filter",
        help="Select the date range for analysis and trends."
    )

    if start_date and end_date and start_date <= end_date:
        filtered_df = health_df[
            (health_df['date'].dt.date >= start_date) &
            (health_df['date'].dt.date <= end_date)
        ].copy()
    else:
        st.sidebar.error("Invalid date range. Please select a start date before or on the end date.")
        filtered_df = health_df.copy() # Default to all data if range is invalid (or handle differently)


    # --- KPIs ---
    clinic_kpis = get_clinic_summary(filtered_df)
    
    st.subheader(f"Operational Metrics ({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')})")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        tt_status = "High" if clinic_kpis['avg_test_turnaround'] > app_config.TARGET_TEST_TURNAROUND_DAYS + 1 else \
                    "Moderate" if clinic_kpis['avg_test_turnaround'] > app_config.TARGET_TEST_TURNAROUND_DAYS else "Low"
        render_kpi_card("Avg. Test Turnaround", f"{clinic_kpis['avg_test_turnaround']:.1f} days", "⏱️", status=tt_status,
                        help_text=f"Target: {app_config.TARGET_TEST_TURNAROUND_DAYS} days. Average time from test sample to result.")
    with kpi_cols[1]:
        ptr_status = "High" if clinic_kpis['positive_test_rate'] > 15 else \
                     "Moderate" if clinic_kpis['positive_test_rate'] > 5 else "Low"
        render_kpi_card("Positive Test Rate", f"{clinic_kpis['positive_test_rate']:.1f}%", "➕", status=ptr_status,
                        help_text="Percentage of lab tests that are positive.")
    with kpi_cols[2]:
        cs_status = "High" if clinic_kpis['critical_supply_items'] > 2 else \
                    "Moderate" if clinic_kpis['critical_supply_items'] > 0 else "Low"
        render_kpi_card("Critical Supply Items", str(clinic_kpis['critical_supply_items']), "📦", status=cs_status,
                        help_text=f"Number of supply items with <{app_config.CRITICAL_SUPPLY_DAYS} days stock remaining.")
    with kpi_cols[3]:
        ptc_status = "Moderate" if clinic_kpis['pending_tests_count'] > 20 else "Low"
        render_kpi_card("Pending Tests", str(clinic_kpis['pending_tests_count']), "🧪", status=ptc_status,
                        help_text="Number of tests currently awaiting results.")
    
    st.markdown("---")
    
    # --- Tabs for Detailed Views ---
    tab_tests, tab_supplies, tab_patients = st.tabs(["🔬 Test Management", "💊 Supply Chain", "🧍 Patient Insights"])

    with tab_tests:
        st.subheader("Test Results & Performance")
        col_test1, col_test2 = st.columns([0.4, 0.6]) # Donut smaller
        
        with col_test1:
            test_results_summary = filtered_df.dropna(subset=['test_result']) \
                                   .groupby('test_result')['patient_id'].nunique().reset_index()
            test_results_summary.columns = ['Test Result', 'Count']
            if not test_results_summary.empty:
                st.plotly_chart(plot_donut_chart(
                    test_results_summary, 'Test Result', 'Count', "Test Result Distribution"
                ), use_container_width=True)
            else:
                st.caption("No test result data for selected period.")
        
        with col_test2:
            turnaround_trend_df = filtered_df.dropna(subset=['test_turnaround_days'])
            turnaround_trend = get_trend_data(turnaround_trend_df, 
                                              'test_turnaround_days', period='D', date_col='test_date') # Use test_date
            if not turnaround_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    turnaround_trend, "Avg. Test Turnaround Time Trend", 
                    y_axis_title="Days", target_line=app_config.TARGET_TEST_TURNAROUND_DAYS, 
                    target_label=f"Target: {app_config.TARGET_TEST_TURNAROUND_DAYS} Days",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else:
                st.caption("No turnaround time data for trend.")

    with tab_supplies:
        st.subheader("Supply Levels & Forecast")
        # Supply forecast should use the full health_df to get latest levels, then forecast from there
        supply_forecast_df = get_supply_forecast_data(health_df) 
        
        if not supply_forecast_df.empty:
            available_items = sorted(supply_forecast_df['item'].unique())
            selected_item = st.selectbox("Select Supply Item for Forecast:", available_items, key="supply_item_select_clinic")

            if selected_item:
                item_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_item].copy()
                # Ensure it's sorted by date if not already for plotting
                item_forecast_df.sort_values('date', inplace=True)
                item_forecast_df.set_index('date', inplace=True)
                
                if not item_forecast_df.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        item_forecast_df['forecast_days'],
                        f"Forecast: {selected_item} (Days of Supply Remaining)",
                        y_axis_title="Days of Supply",
                        target_line=app_config.CRITICAL_SUPPLY_DAYS, 
                        target_label=f"Critical Level ({app_config.CRITICAL_SUPPLY_DAYS} Days)",
                        show_ci=True, 
                        lower_bound_series=item_forecast_df['lower_ci'], 
                        upper_bound_series=item_forecast_df['upper_ci'],
                        height=app_config.DEFAULT_PLOT_HEIGHT + 50 # Slightly taller for CI
                    ), use_container_width=True)
                else:
                    st.info(f"No forecast data available for {selected_item}.")
            else:
                st.info("Select a supply item to view its forecast.")
        else:
            st.caption("No supply data available to generate forecasts.")

    with tab_patients:
        st.subheader("Patient Load & Conditions")
        patient_load_cols = st.columns(2)
        with patient_load_cols[0]:
            # Daily patient visits (unique patients per day)
            daily_visits = filtered_df.groupby(filtered_df['date'].dt.date)['patient_id'].nunique()
            daily_visits.index = pd.to_datetime(daily_visits.index) # Convert index to datetime for plotting
            daily_visits_series = pd.Series(daily_visits.values, index=daily_visits.index, name="Unique Patients")

            if not daily_visits_series.empty:
                 st.plotly_chart(plot_annotated_line_chart(
                    daily_visits_series, "Daily Unique Patient Visits", y_axis_title="Number of Patients",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else:
                st.caption("No patient visit data for trend.")

        with patient_load_cols[1]:
            # Top conditions
            top_conditions = filtered_df['condition'].value_counts().nlargest(5).reset_index()
            top_conditions.columns = ['condition', 'count']
            if not top_conditions.empty:
                st.plotly_chart(plot_bar_chart(
                    top_conditions, x_col='condition', y_col='count',
                    title="Top 5 Conditions (Selected Period)",
                    x_axis_title="Condition", y_axis_title="Number of Cases",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else:
                st.caption("No condition data.")
        
        st.markdown("---")
        st.markdown("###### Flagged Patient Cases for Review")
        # Example: Patients with high risk recently tested positive or have pending critical tests
        flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS['moderate'])
        if not flagged_patients_clinic_df.empty:
            display_cols_clinic = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result', 'referral_status', 'alert_reason']
            actual_cols_clinic = [col for col in display_cols_clinic if col in flagged_patients_clinic_df.columns]
            st.dataframe(
                flagged_patients_clinic_df[actual_cols_clinic].sort_values(by='ai_risk_score', ascending=False).head(10), 
                use_container_width=True,
                column_config={
                     "ai_risk_score": st.column_config.NumberColumn(format="%d")
                }
            )
        else:
            st.info("No specific patient cases flagged for review in the selected period.")
