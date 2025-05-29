# pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_zone_data, load_iot_clinic_environment_data,
    enrich_zone_geodata_with_health_aggregates,
    get_district_summary_kpis, get_trend_data, hash_geodataframe
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_layered_choropleth_map, plot_annotated_line_chart,
    plot_bar_chart, plot_heatmap # plot_heatmap is imported but not used, consider removing if not planned
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None,
    gpd.GeoDataFrame: hash_geodataframe
})
def get_district_page_data():
    logger.info("Getting district page data (including IoT)...")
    health_df = load_health_records()
    zone_base_gdf = load_zone_data() # Should return GDF with CRS
    iot_df = load_iot_clinic_environment_data()

    # Prepare safe empty structures with expected columns if primary loads fail
    expected_health_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score", "avg_daily_steps"] # Added avg_daily_steps
    expected_iot_cols = ['timestamp', 'clinic_id', 'zone_id', 'avg_co2_ppm']
    expected_gdf_cols = ['zone_id', 'name', 'population', 'geometry'] # From load_zone_data

    health_df_safe = health_df if health_df is not None and not health_df.empty else pd.DataFrame(columns=expected_health_cols)
    # Ensure zone_base_gdf_safe has a CRS even if empty
    if zone_base_gdf is not None and not zone_base_gdf.empty:
        zone_base_gdf_safe = zone_base_gdf
    else:
        zone_base_gdf_safe = gpd.GeoDataFrame(columns=expected_gdf_cols, crs=app_config.DEFAULT_CRS) # Use default CRS from config

    iot_df_safe = iot_df if iot_df is not None and not iot_df.empty else pd.DataFrame(columns=expected_iot_cols)

    if health_df_safe.empty and zone_base_gdf_safe.empty: # pragma: no cover
        st.error("🚨 CRITICAL ERROR: Both health records and zone geographic data failed to load or are empty. Dashboard functionality severely limited.")
        logger.critical("Load_health_records returned empty AND load_zone_data returned None/empty.")
        return health_df_safe, zone_base_gdf_safe, iot_df_safe

    if zone_base_gdf_safe.empty : # pragma: no cover
        st.error("🚨 ERROR: Zone geographic data could not be loaded. Map and zonal analysis will be unavailable.")
        logger.error("load_zone_data returned None or empty GeoDataFrame.")
        # Ensure enriched_zone_gdf is at least an empty GDF with CRS
        empty_enriched_gdf = gpd.GeoDataFrame(columns=expected_gdf_cols, crs=app_config.DEFAULT_CRS) # Use default CRS
        return health_df_safe, empty_enriched_gdf, iot_df_safe
    
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_base_gdf_safe,
        health_df_safe,
        iot_df_safe
    )

    if enriched_zone_gdf is None or enriched_zone_gdf.empty: # pragma: no cover
         st.warning("⚠️ Warning: Failed to create enriched zone data. Map/zonal stats may use base zone data or be incomplete.")
         logger.warning("Enrichment of zone GDF resulted in empty or None GDF. Falling back to base_zone_gdf if available.")
         # Fallback to base GDF, ensure it has a CRS if it was somehow lost
         if zone_base_gdf_safe.crs is None and app_config.DEFAULT_CRS:
             zone_base_gdf_safe = zone_base_gdf_safe.set_crs(app_config.DEFAULT_CRS, allow_override=True)
         return health_df_safe, zone_base_gdf_safe, iot_df_safe

    # Ensure enriched_zone_gdf has a CRS
    if enriched_zone_gdf.crs is None and app_config.DEFAULT_CRS:
        enriched_zone_gdf = enriched_zone_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True) # Use default CRS

    logger.info("Successfully retrieved and processed district page data including IoT.")
    return health_df_safe, enriched_zone_gdf, iot_df_safe

health_records_for_trends, district_map_and_zonal_stats_gdf, iot_records_for_trends = get_district_page_data()

# --- Main Page Structure ---
st.title("🗺️ District Health Officer Dashboard")
st.markdown("**Strategic Overview for Population Health (Tijuana Focus: TB, Malaria, HIV, STIs, NTDs etc.) & Environmental Health**")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("District Filters")
start_date_filter, end_date_filter = None, None 
min_date_overall, max_date_overall = None, None
default_start_date_overall = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
default_end_date_overall = pd.Timestamp('today').date()

# Determine date range primarily from health records
if not health_records_for_trends.empty and 'date' in health_records_for_trends.columns:
    if not pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']): # pragma: no cover
        health_records_for_trends['date'] = pd.to_datetime(health_records_for_trends['date'], errors='coerce')
    health_records_for_trends.dropna(subset=['date'], inplace=True)

    if not health_records_for_trends.empty:
        min_date_dt = health_records_for_trends['date'].min()
        max_date_dt = health_records_for_trends['date'].max()

        if pd.notna(min_date_dt) and pd.notna(max_date_dt):
            min_date_overall = min_date_dt.date()
            max_date_overall = max_date_dt.date()
            default_end_date_overall = max_date_overall
            default_start_dt = max_date_dt - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
            default_start_date_overall = default_start_dt.date()
            if default_start_date_overall < min_date_overall: default_start_date_overall = min_date_overall
            if default_start_date_overall > max_date_overall: default_start_date_overall = min_date_overall # Ensure start is not after end
        else: logger.warning("Date range in health records is invalid after NaT drop.") # pragma: no cover
    else: logger.warning("No valid date data in health records for trend filtering after processing.") # pragma: no cover
else: logger.warning("Health records empty or 'date' column missing. Using system default date range.") # pragma: no cover

# Fallback for min/max if not set by health_records
if min_date_overall is None: min_date_overall = default_start_date_overall - pd.Timedelta(days=30) # Generic fallback
if max_date_overall is None: max_date_overall = default_end_date_overall

# Ensure default_start is not after default_end
if default_start_date_overall > default_end_date_overall:
    default_start_date_overall = default_end_date_overall

start_date_filter, end_date_filter = st.sidebar.date_input(
    "Select Date Range for Trends:", value=[default_start_date_overall, default_end_date_overall],
    min_value=min_date_overall, max_value=max_date_overall, key="district_date_filter_final_v3",
    help="This date range applies to time-series trend charts."
)

# Filter health_records for trends
filtered_health_records_for_trends = pd.DataFrame(columns=health_records_for_trends.columns if not health_records_for_trends.empty else None)
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not health_records_for_trends.empty:
    if 'date' in health_records_for_trends.columns and pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']):
        filtered_health_records_for_trends = health_records_for_trends[
            (health_records_for_trends['date'].dt.date >= start_date_filter) &
            (health_records_for_trends['date'].dt.date <= end_date_filter)
        ].copy()
    else: # Should not happen if date processing above is effective
        logger.warning("Health records 'date' column issue post-filter setup. Using all health data.") # pragma: no cover
        filtered_health_records_for_trends = health_records_for_trends.copy()
elif not health_records_for_trends.empty:
    logger.info("Trend date filter invalid or not set, using all available health trend data.")
    filtered_health_records_for_trends = health_records_for_trends.copy()

# Filter IoT records for trends similarly
filtered_iot_records_for_trends = pd.DataFrame(columns=iot_records_for_trends.columns if not iot_records_for_trends.empty else None)
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not iot_records_for_trends.empty:
    if 'timestamp' in iot_records_for_trends.columns:
        if not pd.api.types.is_datetime64_any_dtype(iot_records_for_trends['timestamp']): # pragma: no cover
            iot_records_for_trends['timestamp'] = pd.to_datetime(iot_records_for_trends['timestamp'], errors='coerce')
        iot_records_for_trends.dropna(subset=['timestamp'], inplace=True) # Drop if coerce failed

        if not iot_records_for_trends.empty and pd.api.types.is_datetime64_any_dtype(iot_records_for_trends['timestamp']): # Check again
            filtered_iot_records_for_trends = iot_records_for_trends[
                (iot_records_for_trends['timestamp'].dt.date >= start_date_filter) &
                (iot_records_for_trends['timestamp'].dt.date <= end_date_filter)
            ].copy()
        else:
            logger.warning("IoT records 'timestamp' column issue or all NaT after processing. Using all IoT data.") # pragma: no cover
            filtered_iot_records_for_trends = iot_records_for_trends.copy()
    else:
        logger.warning("IoT records missing 'timestamp' column. Using all IoT data.") # pragma: no cover
        filtered_iot_records_for_trends = iot_records_for_trends.copy()
elif not iot_records_for_trends.empty:
    logger.info("Using all available IoT trend data due to filter issue or no filter set.")
    filtered_iot_records_for_trends = iot_records_for_trends.copy()


# --- KPIs ---
if not district_map_and_zonal_stats_gdf.empty:
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf) # Assumes GDF is not empty
    st.subheader("District-Wide Key Performance Indicators (Overall Aggregates)")
    kpi_cols_1 = st.columns(3)
    with kpi_cols_1[0]: render_kpi_card("Avg. Population Risk", f"{district_kpis.get('avg_population_risk', 0):.1f}", "🎯", status="High" if district_kpis.get('avg_population_risk', 0) > 65 else "Moderate", help_text="Population-weighted average AI risk score.")
    with kpi_cols_1[1]: render_kpi_card("Facility Coverage Score", f"{district_kpis.get('overall_facility_coverage', 0):.1f}%", "🏥", status="Low" if district_kpis.get('overall_facility_coverage', 0) < 60 else "Moderate", help_text="Population-weighted average facility coverage score.")
    # Calculate high_risk_zone_threshold dynamically if district_map_and_zonal_stats_gdf is not empty
    high_risk_zone_threshold = (len(district_map_and_zonal_stats_gdf) * 0.25) if not district_map_and_zonal_stats_gdf.empty else 1
    with kpi_cols_1[2]: render_kpi_card("High-Risk Zones", str(district_kpis.get('zones_high_risk', 0)), "⚠️", status="High" if district_kpis.get('zones_high_risk', 0) > high_risk_zone_threshold else "Moderate", help_text=f"Zones with avg. risk score >= {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    
    st.markdown("##### Key Disease & Environmental Burden")
    kpi_cols_2 = st.columns(3)
    tb_icon = "<img src='https://www.svgrepo.com/show/309948/lungs.svg' width='28' alt='TB' style='vertical-align: middle;'>"
    malaria_icon = "<img src='https://www.svgrepo.com/show/491020/mosquito.svg' width='28' alt='Malaria' style='vertical-align: middle;'>"
    with kpi_cols_2[0]: render_kpi_card("Active TB Cases", str(district_kpis.get('district_tb_burden',0)), tb_icon, icon_is_html=True, status="High" if district_kpis.get('district_tb_burden',0) > 20 else "Moderate")
    with kpi_cols_2[1]: render_kpi_card("Active Malaria Cases", str(district_kpis.get('district_malaria_burden',0)), malaria_icon, icon_is_html=True, status="High" if district_kpis.get('district_malaria_burden',0) > 50 else "Moderate")
    with kpi_cols_2[2]: render_kpi_card("Avg. Clinic CO2", f"{district_kpis.get('avg_zone_co2',0):.0f} ppm", "💨", status="High" if district_kpis.get('avg_zone_co2',0) > app_config.CO2_LEVEL_ALERT_PPM else "Moderate", help_text="District average of zonal avg. CO2 in clinics.")
else: st.warning("Zone data for KPIs is unavailable. Check data loading.") # pragma: no cover
st.markdown("---")

# --- Choropleth Map ---
if not district_map_and_zonal_stats_gdf.empty:
    st.subheader("Interactive Health Map: Tijuana Risk, Disease Burden & Environment")
    map_metric_options = {
        "Average AI Risk Score": "avg_risk_score", "Active TB Cases": "active_tb_cases",
        "Active Malaria Cases": "active_malaria_cases", "HIV Positive Cases": "hiv_positive_cases",
        "Pneumonia Cases": "pneumonia_cases", "Anemia Cases": "anemia_cases", "STI Cases": "sti_cases",
        "Prevalence per 1,000 (Key Infections)": "prevalence_per_1000", 
        "Facility Coverage Score": "facility_coverage_score", "Population": "population",
        "Avg. Patient Steps by Zone": "avg_daily_steps_zone", "Avg. Patient SpO2 by Zone": "avg_spo2_zone",
        "Avg. Zone CO2 (Clinics)": "zone_avg_co2", "Number of Clinics": "num_clinics" # Added num_clinics
    }
    available_map_metrics = {
        k: v for k, v in map_metric_options.items() 
        if v in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v].notna().any()
    }
    
    if available_map_metrics:
        selected_map_metric_display = st.selectbox(
            "Select Metric to Display on Map:", list(available_map_metrics.keys()), 
            key="district_map_metric_final_v2",
            help="Choose a metric to visualize spatially across the district zones."
        )
        selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)

        if selected_map_metric_col:
            color_scale = "OrRd"; higher_is_better_metrics = ["coverage", "socio_economic", "clinics", "steps", "spo2", "num_clinics"] # Added num_clinics
            if any(keyword in selected_map_metric_col.lower() for keyword in higher_is_better_metrics): color_scale = "Mint"
            
            # Define hover columns dynamically based on availability
            base_hover_cols = ['name', 'population']
            dynamic_hover_cols = [selected_map_metric_col]
            if 'num_clinics' in district_map_and_zonal_stats_gdf.columns and selected_map_metric_col != 'num_clinics':
                dynamic_hover_cols.append('num_clinics')
            
            final_hover_cols = base_hover_cols + [col for col in dynamic_hover_cols if col in district_map_and_zonal_stats_gdf.columns]
            final_hover_cols = list(dict.fromkeys(final_hover_cols)) # Remove duplicates

            map_figure = plot_layered_choropleth_map(
                gdf=district_map_and_zonal_stats_gdf, value_col=selected_map_metric_col,
                title=f"{selected_map_metric_display} by Zone",
                id_col='zone_id', featureidkey_prop='properties.zone_id', # Adjusted for typical GeoJSON id access
                color_continuous_scale=color_scale,
                hover_cols=final_hover_cols,
                height=app_config.MAP_PLOT_HEIGHT, center_lat=app_config.TIJUANA_CENTER_LAT,
                center_lon=app_config.TIJUANA_CENTER_LON, zoom_level=app_config.TIJUANA_DEFAULT_ZOOM,
                mapbox_style=app_config.MAPBOX_STYLE # Use config for style
            )
            st.plotly_chart(map_figure, use_container_width=True)
        else:  st.info("Please select a metric to display on the map.") # pragma: no cover
    else: st.warning("No metrics with valid data available for mapping.") # pragma: no cover
else: st.warning("Zone map data is empty. Cannot display map.") # pragma: no cover
st.markdown("---")

# --- Tabs ---
tab_trends, tab_comparison, tab_interventions = st.tabs([
    "📈 District Trends", "📊 Zonal Comparison", "🛠️ Intervention Insights"
])

with tab_trends:
    st.header("📈 District-Wide Health & Environmental Trends")
    if not filtered_health_records_for_trends.empty and start_date_filter and end_date_filter:
        st.markdown(f"Displaying trends from **{start_date_filter.strftime('%d %b %Y')}** to **{end_date_filter.strftime('%d %b %Y')}**.")
        
        st.subheader("Key Disease Trends")
        trend_cols_disease = st.columns(2)
        with trend_cols_disease[0]:
            if 'condition' in filtered_health_records_for_trends.columns and 'patient_id' in filtered_health_records_for_trends.columns:
                tb_trend_df = filtered_health_records_for_trends[filtered_health_records_for_trends['condition'].astype(str) == 'TB'].copy()
                new_tb_cases_trend = get_trend_data(tb_trend_df, 'patient_id', period='W', agg_func='nunique')
                if not new_tb_cases_trend.empty: st.plotly_chart(plot_annotated_line_chart(new_tb_cases_trend, "Weekly New TB Patients Identified", y_axis_title="New TB Patients"), use_container_width=True)
                else: st.caption("No TB trend data for the selected period.")
            else: st.caption("Required columns for TB trend missing.")
        with trend_cols_disease[1]:
            if 'condition' in filtered_health_records_for_trends.columns and 'patient_id' in filtered_health_records_for_trends.columns:
                malaria_trend_df = filtered_health_records_for_trends[filtered_health_records_for_trends['condition'].astype(str) == 'Malaria'].copy()
                new_malaria_cases_trend = get_trend_data(malaria_trend_df, 'patient_id', period='W', agg_func='nunique')
                if not new_malaria_cases_trend.empty: st.plotly_chart(plot_annotated_line_chart(new_malaria_cases_trend, "Weekly New Malaria Patients Identified", y_axis_title="New Malaria Patients"), use_container_width=True)
                else: st.caption("No Malaria trend data for the selected period.")
            else: st.caption("Required columns for Malaria trend missing.")

        st.subheader("Population Wellness & Environmental Trends")
        trend_cols_wellness_env = st.columns(2)
        with trend_cols_wellness_env[0]:
            if 'avg_daily_steps' in filtered_health_records_for_trends.columns:
                avg_steps_trend_dist = get_trend_data(filtered_health_records_for_trends, 'avg_daily_steps', period='W', agg_func='mean')
                if not avg_steps_trend_dist.empty: st.plotly_chart(plot_annotated_line_chart(avg_steps_trend_dist, "Weekly Avg. Patient Steps", y_axis_title="Avg. Steps", target_line=app_config.TARGET_DAILY_STEPS), use_container_width=True)
                else: st.caption("No patient steps trend data for the selected period.")
            else: st.caption("Avg. daily steps data missing.")
        with trend_cols_wellness_env[1]:
            if not filtered_iot_records_for_trends.empty and 'avg_co2_ppm' in filtered_iot_records_for_trends.columns:
                district_avg_co2_trend = get_trend_data(filtered_iot_records_for_trends, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not district_avg_co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(district_avg_co2_trend, "Daily Avg. CO2 (All Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM), use_container_width=True)
                else: st.caption("No CO2 trend data for the selected period.")
            elif filtered_iot_records_for_trends.empty: st.caption("No IoT data available for environmental trends.")
            else: st.caption("Avg. CO2 data missing from IoT records.")
    else:
        st.info("Select a valid date range and ensure data is loaded for trend analysis.")

with tab_comparison:
    st.header("📊 Zonal Comparative Analysis")
    if not district_map_and_zonal_stats_gdf.empty:
        st.markdown("Comparing zones based on overall aggregated health, environmental and resource metrics.")
        comparison_metrics_config_dist = {
            "Avg. AI Risk Score": {"col": "avg_risk_score", "higher_is_worse": True, "format": "{:.1f}"},
            "Active TB Cases": {"col": "active_tb_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "Active Malaria Cases": {"col": "active_malaria_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "HIV Positive Cases": {"col": "hiv_positive_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "Pneumonia Cases": {"col": "pneumonia_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "Anemia Cases": {"col": "anemia_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "STI Cases": {"col": "sti_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "higher_is_worse": True, "format": "{:.1f}"},
            "Facility Coverage Score": {"col": "facility_coverage_score", "higher_is_worse": False, "format": "{:.1f}%"},
            "Population": {"col": "population", "format": "{:,.0f}"}, # higher_is_worse could be False or not set (neutral)
            "Number of Clinics": {"col": "num_clinics", "higher_is_worse": False, "format": "{:.0f}"},
            "Socio-Economic Index": {"col": "socio_economic_index", "higher_is_worse": False, "format": "{:.2f}"},
            "Avg. Patient Steps": {"col": "avg_daily_steps_zone", "higher_is_worse": False, "format":"{:,.0f}"},
            "Avg. Patient SpO2": {"col": "avg_spo2_zone", "higher_is_worse": False, "format":"{:.1f}%"},
            "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "higher_is_worse": True, "format": "{:.0f} ppm"}
        }
        available_comp_metrics_dist = {
            k: v for k,v in comparison_metrics_config_dist.items() 
            if v["col"] in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v["col"]].notna().any()
        }

        if available_comp_metrics_dist:
            st.subheader("Zonal Statistics Overview")
            display_comp_cols_dist = ['name'] + [d['col'] for d in available_comp_metrics_dist.values() if d['col'] != 'name']
            display_comp_cols_dist = list(dict.fromkeys(display_comp_cols_dist)) # Ensure 'name' is first and unique
            
            comparison_df_display_dist = district_map_and_zonal_stats_gdf[display_comp_cols_dist].copy()
            comparison_df_display_dist.rename(columns={'name':'Zone Name'}, inplace=True)
            if 'Zone Name' in comparison_df_display_dist.columns:
                 comparison_df_display_dist.set_index('Zone Name', inplace=True)
            
            style_format_dict_dist = {details["col"]: details["format"] for disp_name, details in available_comp_metrics_dist.items() if "format" in details and details["col"] in comparison_df_display_dist.columns}
            
            # Corrected colormap logic:
            # For "higher is better" (higher_is_worse=False), use 'Greens' (high values are dark green)
            highlight_better_cols = [d["col"] for disp_name, d in available_comp_metrics_dist.items() if not d.get("higher_is_worse", True) and d["col"] in comparison_df_display_dist.columns]
            # For "lower is better" (higher_is_worse=True), use 'Reds' (high values are dark red)
            highlight_worse_cols = [d["col"] for disp_name, d in available_comp_metrics_dist.items() if d.get("higher_is_worse", True) and d["col"] in comparison_df_display_dist.columns]
            
            styler = comparison_df_display_dist.style.format(style_format_dict_dist)
            if highlight_better_cols:
                styler = styler.background_gradient(cmap='Greens', axis=0, subset=pd.IndexSlice[:, highlight_better_cols])
            if highlight_worse_cols:
                styler = styler.background_gradient(cmap='Reds', axis=0, subset=pd.IndexSlice[:, highlight_worse_cols])
            
            st.dataframe(styler, use_container_width=True, height=min(len(comparison_df_display_dist) * 35 + 40, 450))
            
            st.subheader("Visual Comparison by Metric")
            selected_comp_metric_display_dist = st.selectbox("Select metric for bar chart:", list(available_comp_metrics_dist.keys()), key="district_comp_bar_final_v3")
            selected_details_dist = available_comp_metrics_dist.get(selected_comp_metric_display_dist)
            if selected_details_dist:
                selected_col_name_dist = selected_details_dist["col"]
                # If higher_is_worse is False (higher is better), we want to sort descending to see best on top for some charts.
                # However, for a typical bar chart, ascending sort of the metric value might be more natural.
                # Let's keep sort_ascending_bar_dist = (not selected_details_dist.get("higher_is_worse", True))
                # This means if higher is better, it sorts ascending (smallest to largest). If higher is worse, it sorts descending (largest bad value to smallest bad value).
                # Or, for bar charts, often sorting by the value descending is common to see top performers/worst performers first.
                # Let's sort by value descending by default, unless "higher_is_worse" is False (higher is better), then sort ascending.
                # This might be counter-intuitive. Let's simplify: always sort by value for the bar chart. User can visually inspect.
                # For a "top N" or "bottom N" view, specific sorting is key. For general comparison, default sort by x-axis label (name) or by value is fine.
                # plot_bar_chart has `sort_values_by` and `ascending` params.
                
                bar_df_comp_dist = district_map_and_zonal_stats_gdf[['name', selected_col_name_dist]].copy()
                # Default: sort by metric value, descending to show highest values first
                sort_ascending_bar = False # Show highest values at the top/left
                if not selected_details_dist.get("higher_is_worse", True): # If higher is better
                     pass # Still show highest values (best) first by descending sort

                st.plotly_chart(plot_bar_chart(bar_df_comp_dist, x_col='name', y_col=selected_col_name_dist, title=f"{selected_comp_metric_display_dist} by Zone", x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 100, sort_values_by=selected_col_name_dist, ascending=sort_ascending_bar, text_auto = '.2s' if "population" in selected_col_name_dist.lower() else True), use_container_width=True)
        else: st.info("No metrics available for Zonal Comparison.") # pragma: no cover
    else: st.info("No zonal data available for comparison. Check data loading and enrichment.") # pragma: no cover


with tab_interventions:
    st.header("🛠️ Intervention Planning Insights")
    if not district_map_and_zonal_stats_gdf.empty:
        st.markdown("Identify zones based on combined risk factors. Data reflects overall aggregates.")
        # TODO: Consider moving these thresholds to app_config
        FACILITY_COVERAGE_LOW_THRESHOLD = app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD # Example: 50
        TB_BURDEN_HIGH_THRESHOLD = app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD         # Example: 5
        MALARIA_BURDEN_HIGH_THRESHOLD = app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD # Example: 10

        criteria_options_int = {
            f"High Avg. Risk Score (>= {app_config.RISK_THRESHOLDS['district_zone_high_risk']})": "high_risk",
            f"Low Facility Coverage (< {FACILITY_COVERAGE_LOW_THRESHOLD}%)": "low_coverage",
            "High Prevalence (Key Inf., Top 25%)": "high_prevalence",
            f"High TB Burden (> {TB_BURDEN_HIGH_THRESHOLD} active cases)": "high_tb_burden",
            f"High Malaria Burden (> {MALARIA_BURDEN_HIGH_THRESHOLD} active cases)": "high_malaria_burden",
            f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)": "high_clinic_co2"
        }
        
        available_criteria_int = {}
        if 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"High Avg. Risk Score (>= {app_config.RISK_THRESHOLDS['district_zone_high_risk']})"] = "high_risk"
        if 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"Low Facility Coverage (< {FACILITY_COVERAGE_LOW_THRESHOLD}%)"] = "low_coverage"
        if 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns: available_criteria_int["High Prevalence (Key Inf., Top 25%)"] = "high_prevalence"
        if 'active_tb_cases' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"High TB Burden (> {TB_BURDEN_HIGH_THRESHOLD} active cases)"] = "high_tb_burden"
        if 'active_malaria_cases' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"High Malaria Burden (> {MALARIA_BURDEN_HIGH_THRESHOLD} active cases)"] = "high_malaria_burden"
        if 'zone_avg_co2' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)"] = "high_clinic_co2"
        
        if not available_criteria_int: # pragma: no cover
            st.warning("No criteria can be applied due to missing data columns for intervention planning.")
        else:
            selected_criteria_keys_int = st.multiselect(
                "Filter zones by (meets ANY selected criteria):",
                options=list(available_criteria_int.keys()),
                default=list(available_criteria_int.keys())[0:1] if available_criteria_int else [],
                key="intervention_criteria_final_v2"
            )
            priority_masks_int = []
            if not selected_criteria_keys_int: st.info("Select criteria to identify priority zones.")
            else:
                for key_int in selected_criteria_keys_int:
                    criterion_int = available_criteria_int[key_int]
                    if criterion_int == "high_risk": priority_masks_int.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
                    elif criterion_int == "low_coverage": priority_masks_int.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < FACILITY_COVERAGE_LOW_THRESHOLD)
                    elif criterion_int == "high_prevalence" and district_map_and_zonal_stats_gdf['prevalence_per_1000'].notna().any():
                        prev_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
                        if pd.notna(prev_q75): priority_masks_int.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prev_q75)
                    elif criterion_int == "high_tb_burden": priority_masks_int.append(district_map_and_zonal_stats_gdf['active_tb_cases'] > TB_BURDEN_HIGH_THRESHOLD)
                    elif criterion_int == "high_malaria_burden": priority_masks_int.append(district_map_and_zonal_stats_gdf['active_malaria_cases'] > MALARIA_BURDEN_HIGH_THRESHOLD)
                    elif criterion_int == "high_clinic_co2": priority_masks_int.append(district_map_and_zonal_stats_gdf['zone_avg_co2'] > app_config.CO2_LEVEL_IDEAL_PPM)
                
                if priority_masks_int:
                    try:
                        # Align indices before combining boolean Series
                        final_index = district_map_and_zonal_stats_gdf.index
                        aligned_masks = [s.reindex(final_index).fillna(False) for s in priority_masks_int if isinstance(s, pd.Series)]
                        if aligned_masks:
                            final_priority_mask_int = pd.concat(aligned_masks, axis=1).any(axis=1)
                        else: # Should not happen if priority_masks_int is not empty and contains Series
                            final_priority_mask_int = pd.Series([False]*len(district_map_and_zonal_stats_gdf), index=final_index)
                        priority_zones_df_int = district_map_and_zonal_stats_gdf[final_priority_mask_int]
                    except Exception as e_mask_int: logger.error(f"Error applying intervention criteria: {e_mask_int}", exc_info=True); priority_zones_df_int = pd.DataFrame() # pragma: no cover
                    
                    if not priority_zones_df_int.empty:
                        st.markdown("###### Zones Identified for Potential Intervention (Overall Aggregates):")
                        intervention_cols_show = ['name', 'population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2']
                        actual_int_cols = [col for col in intervention_cols_show if col in priority_zones_df_int.columns]
                        
                        sort_by_cols_int = []
                        ascending_int_flags = []
                        if 'avg_risk_score' in actual_int_cols: sort_by_cols_int.append('avg_risk_score'); ascending_int_flags.append(False) # Higher risk first
                        if 'prevalence_per_1000' in actual_int_cols: sort_by_cols_int.append('prevalence_per_1000'); ascending_int_flags.append(False) # Higher prev first
                        if 'zone_avg_co2' in actual_int_cols: sort_by_cols_int.append('zone_avg_co2'); ascending_int_flags.append(False) # Higher CO2 first
                        if 'facility_coverage_score' in actual_int_cols: sort_by_cols_int.append('facility_coverage_score'); ascending_int_flags.append(True) # Lower coverage first

                        priority_zones_df_int_sorted = priority_zones_df_int.sort_values(by=sort_by_cols_int, ascending=ascending_int_flags) if sort_by_cols_int else priority_zones_df_int
                        
                        format_intervention_table = {col: "{:,.0f}" for col in ['population', 'active_tb_cases', 'active_malaria_cases', 'zone_avg_co2']}
                        format_intervention_table.update({col: "{:.1f}" for col in ['avg_risk_score', 'prevalence_per_1000']})
                        if 'facility_coverage_score' in actual_int_cols : format_intervention_table['facility_coverage_score'] = "{:.1f}%"
                        
                        st.dataframe(priority_zones_df_int_sorted[actual_int_cols], use_container_width=True, 
                                     column_config={k:st.column_config.NumberColumn(format=v.replace('%','%%') if isinstance(v, str) else v) for k,v in format_intervention_table.items() if k in actual_int_cols} )
                    else: st.success("✅ No zones meet selected high-priority criteria.")
                elif selected_criteria_keys_int : st.warning("Could not apply selected criteria effectively. Check data columns or criteria definitions.") # pragma: no cover
    else: st.info("No zonal data for intervention insights. Check data loading.") # pragma: no cover
