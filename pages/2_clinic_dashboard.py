# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_dataYou,
    get_clinic_summary, get_clinic_environmental_summary,
    get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Get logger for this page

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st are absolutely right. My apologies for not providing the full `pages/2_clinic_dashboard.py` with the robust date handling included directly in the last response. I will ensure to provide complete files going forward.

Here.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600)
 is the **complete and corrected `pages/2_clinic_dashboard.py`** file, incorporating the more robust date filterdef get_clinic_page_data_extended():
    logger.info("Loading extended data for Clinic Dashboard...")
    health_df = load initialization.

---
**`pages/2_clinic_dashboard.py` (Complete, SME Reviewed and Enhanced)**
---
```python
_health_records()
    iot_df = load_iot_clinic_environment_data()
    logger.info(f"Health records loaded: {not health_df.empty}, IoT data loaded: {not iot_df.# pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
fromempty}")
    return health_df, iot_df

health_df, iot_df_clinic = get_clinic config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data,
    get_clinic_summary, get_clinic_environmental_summary,
    get_page_data_extended()

# --- Main Page ---
if health_df.empty and iot_df_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart
)

st.set_page_config(page_title="Clinic Dashboard - Health Hub",_clinic.empty : # Check if both are problematic
    st.error("🚨 CRITICAL Error: Could not load health records or IoT data. Clinic Dashboard cannot be displayed.")
    st.stop() # Stop if no data at all
else:
    st.title("🏥 Clinic Operations & Environmental Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f Environment**")
    st.markdown("---")

    # --- Sidebar Filters & Date Range Setup ---
    st.sidebar.header("Clinic Filters")
    min_date_cl_page = None
    max_date_cl_page = None
    default_start_cl = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}.")
load_css()

@st.cache_data(ttl=3600)
def get_clinic_page_data_extended():
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    return_RANGE_DAYS_TREND - 1)
    default_end_cl = pd.Timestamp('today').date()

    if not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        if health_df['date'].notna health_df, iot_df

health_df, iot_df_clinic = get_clinic_page_data_extended()

if health_df.empty and (iot_df_clinic is None or iot_df_clinic.empty): # Modified().any():
            min_date_dt_series_cl = health_df['date'].min()
            max_date_dt_series_cl = health_df['date'].max()

            if pd.notna(min_date_dt_series_cl) and pd.notna(max_date_dt_series_cl):
                min_date_ to handle iot_df_clinic being None
    st.error("🚨 CRITICAL Error: Could not load health recordscl_page = min_date_dt_series_cl.date()
                max_date_cl_page = max_date_dt_series_cl.date()
                if min_date_cl_page > max or IoT data. Clinic Dashboard cannot be displayed.")
    st.stop() # Stop if both essential data sources are missing/empty
else:
    st.title("🏥 Clinic Operations & Environmental Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility Environment**")
    st.markdown("---")

    # --- Robust_date_cl_page: min_date_cl_page = max_date_cl_page # Should Date Filter Setup ---
    st.sidebar.header("Clinic Filters")
    min_date_cl_page not happen
                
                # Recalculate defaults based on actual data range
                default_end_cl = max_date_cl_page
                default_start_cl_dt = pd.to_datetime(max_date_cl_page) - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
                default_start_cl = default_start_cl_dt.date()
                if default_start_cl < min_date_cl_page: default_start = None
    max_date_cl_page = None
    # Default to today and N days ago if health_df date processing fails or df is empty
    fallback_max_date_cl = pd.Timestamp('today').date()
    fallback_min_date_cl = fallback_max_date_cl - pd.Timedelta(days=app__cl = min_date_cl_page
                if default_start_cl > default_end_cl: default_start_clconfig.DEFAULT_DATE_RANGE_DAYS_TREND -1) # Match default trend duration
    default_start_cl = default_end_cl # Ensure start is not after end
            else: logger.warning("All 'date' values in health_df are = fallback_min_date_cl
    default_end_cl = fallback_max_date_cl


    if not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        # Ensure date column does not have all NaT values before calling NaT. Using fallback date range for clinic filter.")
        else: logger.warning("Health_df 'date' column has no valid dates. Using fallback date range for clinic filter.")
    else: logger.warning("Health_df empty or ' min/max
        if health_df['date'].notna().any():
            min_date_dt_series = health_df['date'].min()
            max_date_dt_series = health_df['date'].max()

            if pd.notna(min_date_dt_series) and pd.notna(max_date_dt_series):
                min_date_cl_page = min_date_dt_series.date()
                max_date_cl_page = max_date' invalid. Using fallback date range for clinic filter.")
    
    # Final check if min/max still None after logic (e.g., if health_df was completely empty initially)
    if min_date_cl_page is None: min_date_cl_page = pd.Timestamp('today').date() - pd.Timedelta(days=date_dt_series.date()
                
                # Ensure min_date is not after max_date after getting actual data dates
                if min_date_cl_page > max_date_cl_page: min_date_cl_page = max_date_cl_page # pragma: no cover
                
                default_start_cl_dt = pd.to_datetime(max_date_cl_page) - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
                default_start_cl =365) # Wider fallback
    if max_date_cl_page is None: max_date_cl_page = pd.Timestamp('today').date()
    if default_start_cl > default_end_cl: default_start_cl = default_end_cl # Safety

    start_date, end_date = st.sidebar.date_input(
        "Select Date Range for Analysis", 
        value=[default_start_cl, default_end_cl], # Ensure value is always a list of two dates
        min_value=min_date_cl_page, max default_start_cl_dt.date()

                if default_start_cl < min_date_cl_page: default_start_cl = min_date_cl_page
                # Ensure default_start_cl is not after max_date_cl_page
                if default_start_cl > max_date_cl_page : default_start_cl = max_date_cl_page # Can happen if data range < trend days

                default_end_cl = max_date_cl_page # Default end is the latest data date
            else: # All dates were NaT or some other issue
                logger.warning_value=max_date_cl_page, 
        key="clinic_date_range_final_v3", # Unique key
        help="Applies to most charts and KPIs unless specified."
    )
    
    # Filter dataframes based on selected date range
    filtered_df = pd.DataFrame() # Initialize
    if start_date and end_date and start_date <= end_date and not health_df.empty and 'date' in health_df.columns:
        date_mask = (health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)
        filtered_df = health_df[date_mask].copy()
    elif not health_df.empty: # Fallback if date range is("All 'date' values in health_df are NaT for Clinic. Using default fallback date range.")
                min_date_cl_page = fallback_min_date
                max_date_cl_page = fallback_max_date
        else: # pragma: no cover
            logger.warning("Health_df 'date' column contains only NaT values. Using default fallback date range for Clinic.")
            min_date_cl_page = fallback_min_date
            max_date_cl_page = fallback_max_date
    else: # pragma: no cover
        logger.warning("Health_df empty or 'date' column missing/invalid. Using default fallback date range for Clinic.")
        min_date_cl_page = fallback_min_date
        max_date_cl_page = fallback_max_date
        
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range for Analysis", 
        value=[default_start_cl, default_end_cl], # UseYou are absolutely right. My apologies for not providing the fully refined `pages/2_clinic_dashboard.py` in the previous SME review where I detailed the suggestions. I will provide the complete, updated file now, incorporating the robustness checks and refined logic we discussed.

This version assumes:
1.  Your `config/app_config.py` is problematic but data exists
        logger.warning("Date range invalid for health_df filter, using all health_df.")
        filtered_df = health_df.copy()

    filtered_iot_df_clinic = pd.DataFrame() # Initialize
    if start_date and end_date and start_date <= end_date and not iot_df_clinic.empty and 'timestamp' in iot_df_clinic.columns:
        iot_date_mask = (iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)
        filtered_iot_df_clinic = iot_df_clinic[iot_date_mask].copy()
    elif not iot_df_clinic.empty: # Fallback for IoT data
        logger.warning("Date range invalid calculated defaults
        min_value=min_date_cl_page, 
        max_value=max_date_cl_page, 
        key="clinic_date_range_final_v3_complete", # Unique key
        help="Applies to most charts and KPIs unless specified."
    )
    
    # Filter data based on selections, ensuring dataframes are not None before filtering
    filtered_df = pd.DataFrame()
    if start_date and end_date and start_date <= end_date and health_df is not None and not health_df.empty:
        filtered_df = health_df[(health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)].copy()
    elif health_df is not None and not health_df.empty: # Date filter invalid but data exists
        logger.info("Clinic health data: Using full range due to date filter issue or no specific range given by user yet.")
        filtered_df = health_df.copy()

    filtered_iot_df_clinic = pd.DataFrame()
    if start_date and end_date and start_date <= end_date and iot_df_clinic is not None and not iot_df_clinic.empty:
        # Ensure 'timestamp' column exists and up-to-date with all necessary thresholds and settings.
2.  Your `utils/core_data_processing.py` contains all the refined data loading, aggregation, and KPI/alert functions.
3.  Your `utils/ui_visualization_helpers.py` contains the corrected `render_kpi_card` (handling `icon_is_html` properly and using `.strip()`) and robust plotting functions.

---
**`pages/2_clinic_dashboard.py` (Complete, SME Reviewed and Enhanced)**
---
```python
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

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
load_css() for iot_df_clinic filter, using all iot_df_clinic.")
        filtered_iot_df_clinic = iot_df_clinic.copy()

    # --- KPIs ---
    # Disease KPIs
    st.subheader(f"Key Disease Service Metrics ({start_date.strftime('%d %b %Y') if start_date else 'N/A'} - {end_date.strftime('%d %b %Y') if end_date else 'N/A'})")
    if not filtered_df.empty:
        clinic_kpis = get_clinic_summary(filtered_df)
        kpi_cols_disease_cl = st.columns(5)
        with kpi_cols_disease_cl[0]: render_kpi_card("TB Sputum Positivity", f"{clinic_kpis.get('tb_sputum_positivity',0.0):.1f}%", "🔬", status="High" if clinic_kpis.get('tb_sputum_positivity',0) > 10 else "Moderate", help_text="Percentage of sputum tests positive for TB among conclusive tests.")
        with kpi_cols_disease_cl[1]: render_kpi_card("Malaria RDT Positivity", f"{clinic_kpis.get('malaria_rdt_positivity',0.0):.1f}%", "🦟", status="High" if clinic_kpis.get('malaria_rdt_positivity',0) > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Moderate", help_text=f"Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%. RDT positivity for Malaria.")
        with kpi_cols_disease_cl[2]: render_kpi_card("STI Tests Pending", str(clinic_kpis.get('sti_tests_pending',0)), "🧪", status="Moderate" if clinic_kpis.get('sti_tests_pending',0) > 5 else "Low", help_text="Number of key STI tests currently pending results.")
        with kpi_cols_disease_cl[3]: 
            hiv_icon = "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='28' alt='HIV' style='vertical-align: middle;'>"
            render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), hiv_icon, icon_is_html=True, help_text="Number of conclusive HIV tests conducted in the period.")
        with kpi_cols_disease_cl[4]: render_kpi_card("Key Drug Stockouts", str(clinic_kpis.get('critical_disease_supply_items',0)), "💊", status="High" if clinic_kpis.get('critical_disease_supply_items',0) > 0 else "Low", help_text=f"Key disease drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days stock.")
    else:
        st.info("No health record data available for the selected period to display disease service KPIs.")

    # Environmental KPIs
    st.subheader(f"Clinic Environment Snapshot ({start_date.strftime('%d %b %Y') if start_date else 'N/A'} - {end_date.strftime('%d %b %Y') if is datetime before filtering iot_df_clinic
        if 'timestamp' in iot_df_clinic.columns and pd.api.types.is_datetime64_any_dtype(iot_df_clinic['timestamp']):
            filtered_iot_df_clinic = iot_df_clinic[(iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)].copy()
        else: # pragma: no cover
            logger.warning("IoT data missing 'timestamp' column or not datetime. Cannot filter by date.")
            if iot_df_clinic is not None: filtered_iot_df_clinic = iot_df_clinic.copy() # Use unfiltered if timestamp issue
    elif iot_df_clinic is not None and not iot_df_clinic.empty: # Date filter invalid but data exists
        logger.info("Clinic IoT data: Using full range due to date filter issue or no specific range given by user yet.")
        filtered_iot_df_clinic = iot_df_clinic.copy()

    # --- KPIs ---
    clinic_kpis = get_clinic_summary(filtered_df) if not filtered_df.empty else {}
    display_start_date = start_date.strftime('%d %b %Y') if start_date else "N/A"
    display_end_date = end_date.strftime('%d %b %Y') if end_date else "N/A"
    st.subheader(f"Key Disease Service Metrics ({display_start_date} - {display

# --- Data Loading ---
@st.cache_data(ttl=3600)
def get_clinic_page_data_extended():
    logger.info("Loading data for Clinic Dashboard...")
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    if health_df.empty:
        logger.warning("Health records data is empty for Clinic Dashboard.")
    if iot_df.empty:
        logger.info("IoT clinic environment data is empty or not found for Clinic Dashboard.")
    return health_df, iot_df

health_df, iot_df_clinic = get_clinic_page_data_extended()

# --- Main Page ---
if health_df.empty and iot_df_clinic.empty :
    st.error("🚨 CRITICAL Error: Could not load any primary data sources (Health Records or IoT data). Clinic Dashboard cannot be displayed.")
    st.stop() # Stop if no data at all
else:
    st.title("🏥 Clinic Operations & Environmental Dashboard")
    st.markdown("**Monitoring Efficiency, Quality, Resources, and Facility Environment for Improved Patient Care**") # Enhanced subtitle
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("Clinic Dashboard Filters")
    
    # Robust Date Filter Setup
    min_date_cl_page = None
    max_date_cl_page = None
    default_selected_start_date_cl = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1)
    default_selected_end_date_cl = pd.Timestamp('today').date()

    if not health_df.empty and 'date' in health_df.columns and pd.api.types.is_datetime64_any_dtype(health_df['date']):
        if health_df['date'].notna().any():
            min_date_dt_series_cl = health_df['date'].min()
            max_date_dt_series_cl = health_df['date'].max()

            if pd.notna(min_date_dt_series_cl) and pd.notna(max_date_dt_series_cl):
                min_date_cl_page = min_date_dt_series_cl.date()
                max_date_cl_page = max_date_dt_series_cl.date()
                default_selected_end_date_cl = max_date_cl_page # Default end to latest data date

                # Calculate default start date based on the now robust max_date_cl_page
                calculated_default_start = max_date_cl_page - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
                default_selected_start_date_cl = max(calculated_default_start, min_date_cl end_date else 'N/A'})")
    if not filtered_iot_df_clinic.empty:
        env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic)
        kpi_cols_env_cl = st.columns(4)
        with kpi_cols_env_cl[0]: render_kpi_card("Avg. CO2", f"{env_summary_clinic.get('avg_co2',0):.0f} ppm", "💨", status="High" if env_summary_clinic.get('co2_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {env_summary_clinic.get('co2_alert_rooms',0)}")
        with kpi_cols_env_cl[1]: render_kpi_card("Avg. PM2.5", f"{env_summary_clinic.get('avg_pm25',0):.1f} µg/m³", "🌫️", status="High" if env_summary_clinic.get('pm25_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {env_summary_clinic.get('pm25_alert_rooms',0)}")
        with kpi_cols_env_cl[2]: render_kpi_card("Avg. Occupancy", f"{env_summary_clinic.get('avg_occupancy',0):.1f}", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
        with kpi_cols_env_cl[3]: render_kpi_card("Sanitizer Use", f"{env_summary_clinic.get('avg_sanitizer_use_hr',0):.1f}/hr", "🧴", status="Low" if env_summary_clinic.get('avg_sanitizer_use_hr',0) < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High", help_text="Avg. dispenses/hr/unit.")
    else:
        st.info("No IoT environmental data available for the selected period to display environmental KPIs.")
    st.markdown("---")

    # --- Tabs for Detailed Views ---
    tab_tests_cl, tab_supplies_cl, tab_patients_cl, tab_environment_cl = st.tabs([
        "🔬 Disease Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Clinic Environment"
    ])

    with tab_tests_cl:
        st.subheader("Disease-Specific Test Results & Performance")
        if not filtered_df.empty and 'test_type' in filtered_df.columns and 'test_result' in filtered_df.columns:
            col_test1_cl, col_test2_cl = st.columns([0.4, 0.6]) 
            with col_test1_cl:
                tb_tests_df = filtered_df[filtered_df.get('test_type', pd.Series(dtype=str)).astype(str).str.contains("Sputum|GeneXpert", case=False, na=False)].copy()
                if not tb_tests_df.empty:
                    tb_results_summary = tb_tests_df.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
                    tb_results_summary.columns = ['Test Result', 'Count']
                    tb_results_summary = tb_results_summary[~tb_results_summary['Test Result'].isin(['Unknown', 'N/A', 'nan', 'Pending'])]
                    if not tb_results_summary.empty: 
                        st.plotly_chart(plot_donut_chart(tb_results_summary, 'Test Result', 'Count', "TB Test Result Distribution"), use_container_width=True)
                    else: st.caption("No conclusive TB test result data for pie chart.")
                else: st.caption("No TB tests found in selected period.")
            
            with col_test2_cl:
                # General test turnaround trend (could be filtered for specific tests)
                turnaround_trend_df_cl = filtered_df.dropna(subset=['test_turnaround_days', 'test_date']).copy()
                if not turnaround_trend_df_cl.empty and 'test_date' in turnaround_trend_df_cl.columns: # Ensure test_date exists
                    turnaround_trend_df_cl['test_date'] = pd.to_datetime(turnaround_trend_df_cl['test_date'], errors='coerce')
                    turnaround_trend_df_cl.dropna(subset=['test_date'], inplace=True)

                    if not turnaround_trend_df_cl.empty:
                        turnaround_trend_cl = get_trend_data(turnaround_trend_df_cl,'test_turnaround_days', period='D', date_col='test_date')
                        if not turnaround_trend_cl.empty: 
                            st.plotly_chart(plot_annotated_line_chart_end_date})")
    
    kpi_cols_disease_cl = st.columns(5)
    with kpi_cols_disease_cl[0]: render_kpi_card("TB Sputum Positivity", f"{clinic_kpis.get('tb_sputum_positivity',0.0):.1f}%", "🔬", status="High" if clinic_kpis.get('tb_sputum_positivity',0) > 10 else "Moderate", help_text="Percentage of conclusive sputum tests positive for TB.")
    with kpi_cols_disease_cl[1]: render_kpi_card("Malaria RDT Positivity", f"{clinic_kpis.get('malaria_rdt_positivity',0.0):.1f}%", "🦟", status="High" if clinic_kpis.get('malaria_rdt_positivity',0) > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Moderate", help_text=f"Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%. RDT positivity for Malaria.")
    with kpi_cols_disease_cl[2]: render_kpi_card("STI Tests Pending", str(clinic_kpis.get('sti_tests_pending',0)), "🧪", status="Moderate" if clinic_kpis.get('sti_tests_pending',0) > 5 else "Low", help_text="Number of key STI panel tests currently pending results.")
    with kpi_cols_disease_cl[3]: 
        hiv_icon_html_clinic = "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='28' alt='HIV' style='vertical-align: middle;'>"
        render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), hiv_icon_html_clinic, icon_is_html=True, help_text="Number of conclusive HIV tests conducted in period.")
    with kpi_cols_disease_cl[4]: render_kpi_card("Key Drug Stockouts", str(clinic_kpis.get('critical_disease_supply_items',0)), "💊", status="High" if clinic_kpis.get('critical_disease_supply_items',0) > 0 else "Low", help_text=f"Key disease drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days stock.")

    st.subheader(f"Clinic Environment Snapshot (Data from {display_start_date} - {display_end_date})")
    env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic) if not filtered_iot_df_clinic.empty else {}
    kpi_cols_env_cl = st.columns(4) # Adjust if you add/remove env KPIs
    with kpi_cols_env_cl[0]: render_kpi_card("Avg. CO2", f"{env_summary_clinic.get('avg_co2',0):.0f} ppm", "💨", status="High" if env_summary_clinic.get('co2_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {env_summary_clinic.get('co2_alert_rooms',0)}")
    with kpi_cols_env_cl[1]: render_kpi_card("Avg. PM2.5", f"{env_summary_clinic.get('avg_pm25',0):.1f} µg/m³", "🌫️", status="High" if env_summary_clinic.get('pm25_alert_rooms',0) > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {env_summary_clinic.get('pm25_alert_rooms',0)}")
    with kpi_cols_env_cl[2]: render_kpi_card("Avg. Occupancy", f"{env_summary_clinic.get('avg_occupancy',0):.1f} persons", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
    with kpi_cols_env_cl[3]: render_kpi_card("Sanitizer Use /hr", f"{env_summary_clinic.get('avg_sanitizer_use_hr',0):.1f}", "🧴", status="Low" if env_summary_clinic.get('avg_sanitizer_use_hr',0) < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High", help_text="Avg. dispenses per hour per unit.")
    st.markdown("---")

    # --- Tabs for Detailed Views ---
    tab_tests_cl, tab_supplies_cl, tab_patients_cl, tab_environment_cl =_page) # Ensure it's not before min_date

                if default_selected_start_date_cl > default_selected_end_date_cl: # Should only happen if data range is very small
                     default_selected_start_date_cl = default_selected_end_date_cl # Or min_date_cl_page

            else: logger.warning("All 'date' values in health_df are NaT. Using fallback date range for clinic filter.")
        else: logger.warning("Health_df 'date' column is all NaT. Using fallback date range.")
    else: logger.warning("Health_df empty or 'date' column invalid. Using fallback date range for clinic filter.")
    
    # If calculation failed, min_date_cl_page/max_date_cl_page will remain None, so set them to fallbacks
    if min_date_cl_page is None: min_date_cl_page = default_selected_start_date_cl - pd.Timedelta(days=1) # ensure min is before default start
    if max_date_cl_page is None: max_date_cl_page = default_selected_end_date_cl


    start_date, end_date = st.sidebar.date_input(
        "Select Date Range for Analysis:", # Clearer label
        value=[default_selected_start_date_cl, default_selected_end_date_cl],
        min_value=min_date_cl_page,
        max_value=max_date_cl_page,
        key="clinic_date_range_final_v3", # Unique key
        help="This date range applies to most charts and KPIs unless specified."
    )
    
    # Initialize filtered DataFrames
    filtered_df = pd.DataFrame(columns=health_df.columns if not health_df.empty else None)
    filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic.columns if not iot_df_clinic.empty else None)

    if start_date and end_date and start_date <= end_date:
        if not health_df.empty and 'date' in health_df.columns:
            filtered_df = health_df[(health_df['date'].dt.date >= start_date) & (health_df['date'].dt.date <= end_date)].copy()
        
        if not iot_df_clinic.empty and 'timestamp' in iot_df_clinic.columns:
            filtered_iot_df_clinic = iot_df_clinic[(iot_df_clinic['timestamp'].dt.date >= start_date) & (iot_df_clinic['timestamp'].dt.date <= end_date)].copy()
    else: # Fallback if date range is invalid or data is initially empty
        logger.warning("Date filter invalid or source data empty for Clinic. Using full available data (if any) for filtered DFs.")
        if not health_df.empty: filtered_df = health_df.copy()
        if not iot_df_clinic.empty: filtered_iot_df_clinic = iot_df_clinic.copy()

    # --- KPIs ---
    # Ensure filtered_df is passed, and functions handle it if empty
    clinic_kpis = get_clinic_summary(filtered_df) if not filtered_df.empty else get_clinic_summary(pd.DataFrame()) # Pass empty if filtered is empty
    date_range_label = f"({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')})" if start_date and end_date else "(Full Period)"
    
    st.subheader(f"Key Disease Service Metrics {date_range_label}")
    kpi_cols_disease_cl = st.columns(5)
    # Using .get() with default 0.0 for numeric, str() for string display, explicit checks for status
    with kpi_cols_disease_cl[0]:
        tb_pos_val = clinic_kpis.get('tb_sputum_positivity', 0.0)
        render_kpi_card("TB Sputum Positivity", f"{tb_pos_val:.1f}%", "🔬", 
                        status="High" if tb_pos_val > 10 else "Moderate" if tb_pos_val > 0 else "Low",
                        help_text="Percentage of sputum tests (AFB/GeneXpert) positive for TB.")
    with kpi_cols_disease_cl[1]:
        mal_pos_val = clinic_kpis.get('malaria_rdt_positivity', 0.0)
        render_kpi_card("Malaria RDT Positivity", f"{mal_pos_val:.1f}%", "🦟", 
                        status="High" if mal_pos_val > app_config.TARGET_MALARIA_POSITIVITY_RATE + 2 else "Moderate" if mal_pos_val > 0 else "Low", # Adjusted threshold logic slightly
                        help_text=f"Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%. RDT positivity for Malaria.")
    with kpi_cols_disease_cl[2]:
        sti_pend_val = clinic_kpis.get('sti_tests_pending', 0)
        render_kpi_card("STI Tests Pending", str(sti_pend_val), "🧪", 
                        status="Moderate" if sti_pend_val > 5 else "Low",
                        help_text="Number of key STI diagnostic tests currently pending results.")
    with kpi_cols_disease_cl[3]: 
        hiv_icon = "<img src='https://www.svgrepo.com/show/371614/hiv-ribbon.svg' width='28' alt='HIV' style='vertical-align: middle;'>"
        render_kpi_card("HIV Tests This Period", str(clinic_kpis.get('hiv_tests_done_period',0)), hiv_icon, icon_is_html=True, 
                        help_text="Number of conclusive HIV tests conducted in the selected period.")
    with kpi_cols_disease_cl[4]: 
        stock_val = clinic_kpis.get('critical_disease_supply_items',0)
        render_kpi_card("Key Drug Stockouts", str(stock_val), "💊", 
                        status="High" if stock_val > 0 else "Low",
                        help_text=f"Key disease drugs with less than {app_config.CRITICAL_SUPPLY_DAYS} days of stock.")

    st.subheader(f"Clinic Environment Snapshot {date_range_label}")
    env_summary_clinic = get_clinic_environmental_summary(filtered_iot_df_clinic) if not filtered_iot_df_clinic.empty else get_clinic_environmental_summary(pd.DataFrame())
    kpi_cols_env_cl = st.columns(4)
    # ... (Environmental KPIs as before, using .get() and robust defaults) ...
    with kpi_cols_env_cl[0]: co2_val = env_summary_clinic.get('avg_co2',0); co2_rooms = env_summary_clinic.get('co2_alert_rooms',0); render_kpi_card("Avg. CO2", f"{co2_val:.0f} ppm", "💨", status="High" if co2_rooms > 0 else "Low", help_text=f"Rooms w/ CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm: {co2_rooms}")
    with kpi_cols_env_cl[1]: pm25_val = env_summary_clinic.get('avg_pm25',0); pm25_rooms = env_summary_clinic.get('pm25_alert_rooms',0); render_kpi_card("Avg. PM2.5", f"{pm25_val:.1f} µg/m³", "🌫️", status="High" if pm25_rooms > 0 else "Low", help_text=f"Rooms w/ PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³: {pm25_rooms}")
    with kpi_cols_env_cl[2]: occupancy_val = env_summary_clinic.get('avg_occupancy',0); render_kpi_card("Avg. Occupancy", f"{occupancy_val:.1f}", "👨‍👩‍👧‍👦", status="High" if env_summary_clinic.get('high_occupancy_alert') else "Low", help_text=f"Avg. waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
    with kpi_cols_env_cl[3]: sanitizer_val = env_summary_clinic.get('avg_sanitizer_use_hr',0); render_kpi_card("Sanitizer Use", f"{sanitizer_val:.1f}/hr", "🧴", status="Low" if sanitizer_val < app_config.TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER else "High", help_text="Avg. dispenses/hr/unit.")
    st.markdown("---")

    tab_tests_cl, tab_supplies_cl, tab_patients_cl, tab_environment_cl = st.tabs(["🔬 Disease Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Clinic Environment"])

    with tab_tests_cl:
        (
                                turnaround_trend_cl, "Daily Avg. Test Turnaround Time", 
                                y_axis_title="Days", target_line=app_config.TARGET_TEST_TURNAROUND_DAYS,
                                height=app_config.DEFAULT_PLOT_HEIGHT
                            ), use_container_width=True)
                        else: st.caption("No aggregated turnaround time data for trend.")
                    else: st.caption("No valid raw turnaround time data for trend (after date processing).")
                else: st.caption("No raw turnaround time data or 'test_date' column for trend.")
        else: 
            st.info("Health records are empty or missing 'test_type'/'test_result' columns for this tab.")


    with tab_supplies_cl:
        st.subheader("Supply Levels & Forecast (Key Disease Drugs)")
        # Supply forecast generally uses all historical data to find the latest stock level
        supply_forecast_df_all_items = get_supply_forecast_data(health_df) 
        if not supply_forecast_df_all_items.empty:
            key_drug_substrings = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Benznidazole', 'ORS', 'Oxygen', 'Metronidazole', 'Gardasil'] # From config
            available_key_items_cl = sorted([
                item for item in supply_forecast_df_all_items['item'].unique() 
                if pd.notna(item) and any(sub.lower() in str(item).lower() for sub in key_drug_substrings)
            ])
            if not available_key_items_cl: 
                st.info("No forecast data available for key disease drugs based on current stock.")
            else:
                selected_item_clinic_tab = st.selectbox("Select Key Drug for Forecast:", available_key_items_cl, key="supply_item_select_clinic_final_v2")
                if selected_item_clinic_tab: 
                    item_forecast_df_cl = supply_forecast_df_all_items[supply_forecast_df_all_items['item'] == selected_item_clinic_tab].copy()
                    if not item_forecast_df_cl.empty:
                        item_forecast_df_cl.sort_values('date', inplace=True)
                        item_forecast_df_cl.set_index('date', inplace=True)
                        st.plotly_chart(plot_annotated_line_chart(
                            item_forecast_df_cl['forecast_days'],f"Forecast: {selected_item_clinic_tab} (Days of Supply Remaining)",
                            y_axis_title="Days of Supply",target_line=app_config.CRITICAL_SUPPLY_DAYS, 
                            target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)",show_ci=True, 
                            lower_bound_series=item_forecast_df_cl['lower_ci'], 
                            upper_bound_series=item_forecast_df_cl['upper_ci'],
                            height=app_config.DEFAULT_PLOT_HEIGHT + 50
                        ), use_container_width=True)
                    else: st.info(f"No forecast data found for {selected_item_clinic_tab}.") # Should not happen if in available_key_items_cl
        else: st.caption("No overall supply data available to generate forecasts.")


    with tab_patients_cl:
        st.subheader("Patient Load by Key Conditions")
        if not filtered_df.empty and 'condition' in filtered_df.columns and 'date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
            key_conditions_clinic_df = filtered_df[filtered_df['condition'].astype(str).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)]
            if not key_conditions_clinic_df.empty:
                # Ensure 'date' is present for Grouper
                if 'date' not in key_conditions_clinic_df.columns or not pd.api.types.is_datetime64_any_dtype(key_conditions_clinic_df['date']):
                     st.caption("Date column missing or invalid for patient condition summary.") # pragma: no cover
                else:
                    patient_condition_summary = key_conditions_clinic_df.groupby([pd.Grouper(key='date', freq='D'), 'condition'])['patient_id'].nunique().reset_index()
                    patient_condition_summary.rename(columns={'patient_id': 'patient_count'}, inplace=True)
                    if not patient_condition_summary.empty:
                        patient_pivot = patient_condition_summary.pivot_table(index='date', columns='condition', values='patient_count', fill_value=0).reset_index()
                        patient_melt = patient_pivot.melt(id_vars='date', var_name='condition', value_name='patient_count')
                        st.plotly_chart(plot_bar_chart(
                            patient_melt, x_col='date', y_col='patient_count', 
                            title="Daily Patient Count by Key Condition", color_col='condition', 
                            barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT+50
                        ), use_container_width=True)
                    else: st.caption("No patient data for key conditions in selected period to display chart.")
            else: st.caption("No patients with key conditions found in the selected period.")
        else: st.info("Health records are empty or missing 'condition'/'date' columns for patient load analysis.")
        
        st.markdown("---"); st.markdown("###### Flagged Patient Cases for Review (Focused Diseases)")
        flagged_patients_clinic_df = get_patient_alerts_for_clinic(filtered_df, risk_threshold=app_config.RISK_THRESHOLDS['moderate']) if not filtered_df.empty else pd.DataFrame()
        if not flagged_patients_clinic_df.empty:
            # Further filter for KEY_CONDITIONS if desired, or ensure get_patient_alerts_for_clinic handles this focus
            flagged_focused_df = flagged_patients_clinic_df[flagged_patients_clinic_df.get('condition', pd.Series(dtype=str)).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)]
            if not flagged_focused_df.empty:
                st.dataframe(flagged_focused_df[['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result']].sort_values(by='ai_risk_score', ascending=False).head(10), 
                             use_container_width=True, column_config={ "ai_risk_score": st.column_config.NumberColumn(format="%d")})
            else: st.info("No flagged patients with key diseases found in the selected period.")
        else: st.info("No specific patient cases flagged for review in the selected period.")


    with tab_environment_cl:
        st.subheader("Clinic Environmental Monitoring")
        if not filtered_iot_df_clinic.empty:
            # env_summary_clinic already calculated at top of page and used for KPIs
            st.markdown(f"**Alerts Summary (Latest readings in period):** {env_summary_clinic.get('co2_alert_rooms',0)} room(s) with high CO2; {env_summary_clinic.get('pm25_alert_rooms',0)} room(s) with high PM2.5.")
            if env_summary_clinic.get('high_occupancy_alert'): st.warning(f"High waiting room occupancy detected (above {app_config.TARGET_WAITING_ROOM_OCCUPANCY}). Consider patient flow adjustments.")
            
            env_trend_cols_cl_tab = st.columns(2)
            with env_trend_cols_cl_tab[0]:
                co2_trend_cl_tab = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not co2_trend_cl_tab.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend_cl_tab, "Hourly Avg. CO2 Levels", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
                else: st.caption("No CO2 trend data.")
            with env_trend_cols_cl_tab[1]:
                occupancy_trend_cl_tab = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not occupancy_trend_cl_tab.empty: st.plotly_chart(plot_annotated_line_chart(occupancy_trend_cl_tab, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, height=app_config.DEFAULT_PLOT_HEIGHT-20), use_container_width=True)
                else: st.caption("No occupancy trend.")
            
            st.subheader("Latest Room Readings (in selected period)")
            # Ensure required columns for display exist, select only them.
            latest_room_cols = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'waiting_room_occupancy']
            actual_latest_cols = [col for col in latest_room_cols if col in filtered_iot_df_clinic.columns]

            if actual_latest_cols and 'timestamp' in actual_latest_cols and 'clinic_id' in actual_latest_cols and 'room_name' in actual_latest_cols :
                latest_room_readings_cl = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
                if not latest_room_readings_cl.empty: 
                    st.dataframe(latest_room_readings_cl[actual_latest_cols].tail(10), use_container_width=True, height=280) # Show last 10 distinct room readings
                else: st.caption("No detailed room readings for selected period.")
            else: st.caption("Essential columns missing for detailed room readings.")
        else: st.info("No clinic environmental data available for the selected period.")
