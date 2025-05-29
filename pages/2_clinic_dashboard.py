# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_clinic_summary,
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
    
    # Ensure date column is datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(health_df['date']):
        health_df['date'] = pd.to_datetime(health_df['date'], errors='coerce')
        health_df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed

    if health_df.empty: # Check again after potential dropna
        st.error("No valid date entries found in health records after conversion. Cannot proceed with date filters.")
        st.stop()

    min_date = health_df['date'].min().date() 
    max_date = health_df['date'].max().date() 
    
    # Calculate default start date and ensure it's a date object
    default_start_date_dt = pd.to_datetime(max_date) - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
    default_start_date = default_start_date_dt.date()

    # Ensure default_start_date is not before min_date
    if default_start_date < min_date:
        default_start_date = min_date

    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [default_start_date, max_date], # <<<<<<<<<<<< CORRECTED HERE
        min_value=min_date,
        max_value=max_date,
        key="clinic_date_range_filter",
        help="Select the date range for analysis and trends."
    )

    if start_date and end_date and start_date <= end_date:
        # Convert start_date and end_date (which are datetime.date) to datetime for comparison if 'date' column is datetime64
        # Or ensure 'date' column is also just date for direct comparison
        filtered_df = health_df[
            (health_df['date'].dt.date >= start_date) &
            (health_df['date'].dt.date <= end_date)
        ].copy()
    else:
        # This case should ideally not be hit if min_value/max_value and default value are correct
        # But good to have a fallback.
        st.sidebar.error("Invalid date range selected. Defaulting to full range or last 30 days if possible.")
        # If start_date or end_date is None (can happen if user clears one), handle it
        if start_date and end_date: # Both are not None
             filtered_df = health_df[
                (health_df['date'].dt.date >= default_start_date) &
                (health_df['date'].dt.date <= max_date)
            ].copy()
        else: # One or both are None, default to something safe
            filtered_df = health_df[health_df['date'].dt.date >= default_start_date].copy()


    # --- KPIs ---
    # (rest of your clinic dashboard code...)
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
            # Filter out 'Unknown' or 'N/A' if they are not meaningful for this chart
            test_results_summary = test_results_summary[~test_results_summary['Test Result'].isin(['Unknown', 'N/A'])]
            if not test_results_summary.empty:
                st.plotly_chart(plot_donut_chart(
                    test_results_summary, 'Test Result', 'Count', "Test Result Distribution"
                ), use_container_width=True)
            else:
                st.caption("No conclusive test result data for selected period.")
        
        with col_test2:
            turnaround_trend_df = filtered_df.dropna(subset=['test_turnaround_days', 'test_date'])
            # Ensure test_date is datetime for Grouper
            turnaround_trend_df['test_date'] = pd.to_datetime(turnaround_trend_df['test_date'], errors='coerce')
            turnaround_trend_df.dropna(subset=['test_date'], inplace=True)

            if not turnaround_trend_df.empty:
                turnaround_trend = get_trend_data(turnaround_trend_df, 
                                                'test_turnaround_days', period='D', date_col='test_date')
                if not turnaround_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        turnaround_trend, "Avg. Test Turnaround Time Trend", 
                        y_axis_title="Days", target_line=app_config.TARGET_TEST_TURNAROUND_DAYS, 
                        target_label=f"Target: {app_config.TARGET_TEST_TURNAROUND_DAYS} Days",
                        height=app_config.DEFAULT_PLOT_HEIGHT
                    ), use_container_width=True)
                else:
                    st.caption("No aggregated turnaround time data for trend.")
            else:
                st.caption("No raw turnaround time data for trend.")


    with tab_supplies:
        st.subheader("Supply Levels & Forecast")
        # Supply forecast uses the full health_df to get latest levels, then forecasts from there
        supply_forecast_df_all_items = get_supply_forecast_data(health_df) 
        
        if not supply_forecast_df_all_items.empty:
            available_items = sorted(supply_forecast_df_all_items['item'].unique())
            # Filter out 'Unknown' or 'nan' if they appear as items
            available_items = [item for item in available_items if pd.notna(item) and str(item).lower() not in ['unknown', 'nan', 'na', 'n/a']]

            if not available_items:
                 st.info("No valid supply items found for forecasting after filtering.")
            else:
                selected_item = st.selectbox("Select Supply Item for Forecast:", available_items, key="supply_item_select_clinic")

                if selected_item:
                    item_forecast_df = supply_forecast_df_all_items[supply_forecast_df_all_items['item'] == selected_item].copy()
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
                            height=app_config.DEFAULT_PLOT_HEIGHT + 50
                        ), use_container_width=True)
                    else:
                        st.info(f"No forecast data available for {selected_item}.")
        else:
            st.caption("No supply data available to generate forecasts.")

    with tab_patients:
        st.subheader("Patient Load & Conditions")
        patient_load_cols = st.columns(2)
        with patient_load_cols[0]:
            daily_visits = filtered_df.groupby(filtered_df['date'].dt.date)['patient_id'].nunique()
            if not daily_visits.empty:
                daily_visits.index = pd.to_datetime(daily_visits.index) 
                daily_visits_series = pd.Series(daily_visits.values, index=daily_visits.index, name="Unique Patients")
                st.plotly_chart(plot_annotated_line_chart(
                    daily_visits_series, "Daily Unique Patient Visits", y_axis_title="Number of Patients",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else:
                st.caption("No patient visit data for trend.")

        with patient_load_cols[1]:
            # Filter out 'Unknown' or 'N/A' conditions before counting
            valid_conditions_df = filtered_df[~filtered_df['condition'].isin(['Unknown', 'N/A', 'nan'])]
            if not valid_conditions_df.empty:
                top_conditions = valid_conditions_df['condition'].value_counts().nlargest(5).reset_index()
                top_conditions.columns = ['condition', 'count']
                if not top_conditions.empty:
                    st.plotly_chart(plot_bar_chart(
                        top_conditions, x_col='condition', y_col='count',
                        title="Top 5 Conditions (Selected Period)",
                        x_axis_title="Condition", y_axis_title="Number of Cases",
                        height=app_config.DEFAULT_PLOT_HEIGHT,
                        sort_values_by='count', ascending=False # Sort bars
                    ), use_container_width=True)
                else:
                    st.caption("No significant condition data.")
            else:
                st.caption("No valid condition data available.")
        
        st.markdown("---")
        st.markdown("###### Flagged Patient Cases for Review")
        flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS['moderate'])
        if not flagged_patients_clinic_df.empty:
            display_cols_clinic = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result', 'referral_status', 'alert_reason']
            actual_cols_clinic = [col for col in display_cols_clinic if col in flagged_patients_clinic_df.columns]
            st.dataframe(
                flagged_patients_clinic_df[actual_cols_clinic].sort_values(by='ai_risk_score', ascending=False).head(10), 
                use_container_width=True,
                column_config={ "ai_risk_score": st.column_config.NumberColumn(format="%d") }
            )
        else:
            st.info("No specific patient cases flagged for review in the selected period.")
