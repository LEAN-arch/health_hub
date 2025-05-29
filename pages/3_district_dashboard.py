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
    plot_bar_chart, plot_heatmap
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
    zone_base_gdf = load_zone_data()
    iot_df = load_iot_clinic_environment_data()

    # Prepare safe empty structures with expected columns if primary loads fail
    expected_health_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score"] # Add more if needed by enrich
    expected_iot_cols = ['timestamp', 'clinic_id', 'zone_id', 'avg_co2_ppm'] # Add more if needed
    expected_gdf_cols = ['zone_id', 'name', 'population', 'geometry'] # From load_zone_data

    health_df_safe = health_df if health_df is not None and not health_df.empty else pd.DataFrame(columns=expected_health_cols)
    zone_base_gdf_safe = zone_base_gdf if zone_base_gdf is not None and not zone_base_gdf.empty else gpd.GeoDataFrame(columns=expected_gdf_cols, crs="EPSG:4326")
    iot_df_safe = iot_df if iot_df is not None and not iot_df.empty else pd.DataFrame(columns=expected_iot_cols)


    if health_df_safe.empty and zone_base_gdf_safe.empty: # pragma: no cover
        st.error("🚨 CRITICAL ERROR: Both health records and zone geographic data failed to load or are empty. Dashboard functionality severely limited.")
        logger.critical("Load_health_records returned empty AND load_zone_data returned None/empty.")
        # Return structures that won't break downstream, but will show no data
        return health_df_safe, zone_base_gdf_safe, iot_df_safe

    if zone_base_gdf_safe.empty : # pragma: no cover
        st.error("🚨 ERROR: Zone geographic data could not be loaded. Map and zonal analysis will be unavailable.")
        logger.error("load_zone_data returned None or empty GeoDataFrame.")
        # We can still proceed with health_df for district-wide trends, but GDF dependent parts will be empty
        # Ensure enriched_zone_gdf is at least an empty GDF
        return health_df_safe, gpd.GeoDataFrame(columns=expected_gdf_cols, crs="EPSG:4326"), iot_df_safe
    
    # Enrich the zone_base_gdf with aggregated health data and IoT data
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_base_gdf_safe,
        health_df_safe, # Pass even if empty, enrich function handles it
        iot_df_safe     # Pass even if empty, enrich function handles it
    )

    if enriched_zone_gdf is None or enriched_zone_gdf.empty: # pragma: no cover
         st.warning("⚠️ Warning: Failed to create enriched zone data. Map/zonal stats may use base zone data or be incomplete.")
         logger.warning("Enrichment of zone GDF resulted in empty or None GDF. Falling back to base_zone_gdf if available.")
         return health_df_safe, zone_base_gdf_safe, iot_df_safe # Fallback

    logger.info("Successfully retrieved and processed district page data including IoT.")
    return health_df_safe, enriched_zone_gdf, iot_df_safe

health_records_for_trends, district_map_and_zonal_stats_gdf, iot_records_for_trends = get_district_page_data()

# --- Main Page Structure ---
st.title("🗺️ District Health Officer Dashboard")
st.markdown("**Strategic Overview for Population Health (Tijuana Focus: TB, Malaria, HIV, STIs, NTDs etc.) & Environmental Health**")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("District Filters")
start_date_filter, end_date_filter = None, None # Initialize

if not health_records_for_trends.empty and 'date' in health_records_for_trends.columns:
    # Ensure date column is datetime type and not all NaT
    if not pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']): # pragma: no cover
        health_records_for_trends['date'] = pd.to_datetime(health_records_for_trends['date'], errors='coerce')
    health_records_for_trends.dropna(subset=['date'], inplace=True) # Drop if date couldn't be parsed

    if not health_records_for_trends.empty: # Check again
        min_date_dt = health_records_for_trends['date'].min()
        max_date_dt = health_records_for_trends['date'].max()

        if pd.notna(min_date_dt) and pd.notna(max_date_dt):
            min_date = min_date_dt.date()
            max_date = max_date_dt.date()
            default_start_dt = max_date_dt - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
            default_start_date = default_start_dt.date()
            if default_start_date < min_date: default_start_date = min_date
            if default_start_date > max_date: default_start_date = min_date if min_date <= max_date else max_date

            start_date_filter, end_date_filter = st.sidebar.date_input(
                "Select Date Range for Trends:", value=[default_start_date, max_date],
                min_value=min_date, max_value=max_date, key="district_date_filter_final_v3", # Unique key
                help="This date range applies to time-series trend charts."
            )
        else: st.sidebar.warning("Date range in health records is invalid.") # pragma: no cover
    else: st.sidebar.warning("No valid date data for trend filtering after processing.") # pragma: no cover
else: st.sidebar.warning("Health records empty or 'date' column missing. Cannot set trend date filter.") # pragma: no cover

# Filter health_records for trends
filtered_health_records_for_trends = pd.DataFrame() # Initialize as empty
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not health_records_for_trends.empty:
    filtered_health_records_for_trends = health_records_for_trends[
        (health_records_for_trends['date'].dt.date >= start_date_filter) &
        (health_records_for_trends['date'].dt.date <= end_date_filter)
    ].copy()
elif not health_records_for_trends.empty: # Date filter invalid but data exists
    logger.info("Trend date filter invalid or not set, using all available trend data.")
    filtered_health_records_for_trends = health_records_for_trends.copy()

# Filter IoT records for trends similarly
filtered_iot_records_for_trends = pd.DataFrame()
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not iot_records_for_trends.empty:
    filtered_iot_records_for_trends = iot_records_for_trends[
        (iot_records_for_trends['timestamp'].dt.date >= start_date_filter) &
        (iot_records_for_trends['timestamp'].dt.date <= end_date_filter)
    ].copy()
elif not iot_records_for_trends.empty:
    logger.info("Using all available IoT trend data due to filter issue or no filter set.")
    filtered_iot_records_for_trends = iot_records_for_trends.copy()


# --- KPIs ---
if not district_map_and_zonal_stats_gdf.empty:
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf)
    st.subheader("District-Wide Key Performance Indicators (Overall Aggregates)")
    kpi_cols_1 = st.columns(3)
    with kpi_cols_1[0]: render_kpi_card("Avg. Population Risk", f"{district_kpis.get('avg_population_risk', 0):.1f}", "🎯", status="High" if district_kpis.get('avg_population_risk', 0) > 65 else "Moderate", help_text="Population-weighted average AI risk score.")
    with kpi_cols_1[1]: render_kpi_card("Facility Coverage Score", f"{district_kpis.get('overall_facility_coverage', 0):.1f}%", "🏥", status="Low" if district_kpis.get('overall_facility_coverage', 0) < 60 else "Moderate", help_text="Population-weighted average facility coverage score.")
    with kpi_cols_1[2]: render_kpi_card("High-Risk Zones", str(district_kpis.get('zones_high_risk', 0)), "⚠️", status="High" if district_kpis.get('zones_high_risk', 0) > (len(district_map_and_zonal_stats_gdf) * 0.25) else "Moderate", help_text=f"Zones with avg. risk score >= {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    
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
        "Avg. Zone CO2 (Clinics)": "zone_avg_co2"
    }
    available_map_metrics = {
        k: v for k, v in map_metric_options.items() 
        if v in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v].notna().any()
    }
    
    if available_map_metrics:
        selected_map_metric_display = st.selectbox(
            "Select Metric to Display on Map:", list(available_map_metrics.keys()), 
            key="district_map_metric_final_v2", # Unique key
            help="Choose a metric to visualize spatially across the district zones."
        )
        selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)

        if selected_map_metric_col:
            color_scale = "OrRd"; higher_is_better_metrics = ["coverage", "socio_economic", "clinics", "steps", "spo2"]
            if any(keyword in selected_map_metric_col.lower() for keyword in higher_is_better_metrics): color_scale = "Mint"
            
            map_figure = plot_layered_choropleth_map(
                gdf=district_map_and_zonal_stats_gdf, value_col=selected_map_metric_col,
                title=f"{selected_map_metric_display} by Zone",
                id_col='zone_id', featureidkey_prop='zone_id', color_continuous_scale=color_scale,
                hover_cols=['name', 'population', selected_map_metric_col, 'num_clinics'], # Keep hover simple for now
                height=app_config.MAP_PLOT_HEIGHT, center_lat=app_config.TIJUANA_CENTER_LAT,
                center_lon=app_config.TIJUANA_CENTER_LON, zoom_level=app_config.TIJUANA_DEFAULT_ZOOM,
                mapbox_style="open-street-map" # Start with OSM for max compatibility
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
            tb_trend_df = filtered_health_records_for_trends[filtered_health_records_for_trends['condition'].astype(str) == 'TB'].copy()
            new_tb_cases_trend = get_trend_data(tb_trend_df, 'patient_id', period='W', agg_func='nunique') # Weekly new unique TB patients
            if not new_tb_cases_trend.empty: st.plotly_chart(plot_annotated_line_chart(new_tb_cases_trend, "Weekly New TB Patients Identified", y_axis_title="New TB Patients"), use_container_width=True)
            else: st.caption("No TB trend data.")
        with trend_cols_disease[1]:
            malaria_trend_df = filtered_health_records_for_trends[filtered_health_records_for_trends['condition'].astype(str) == 'Malaria'].copy()
            new_malaria_cases_trend = get_trend_data(malaria_trend_df, 'patient_id', period='W', agg_func='nunique')
            if not new_malaria_cases_trend.empty: st.plotly_chart(plot_annotated_line_chart(new_malaria_cases_trend, "Weekly New Malaria Patients Identified", y_axis_title="New Malaria Patients"), use_container_width=True)
            else: st.caption("No Malaria trend data.")

        st.subheader("Population Wellness & Environmental Trends")
        trend_cols_wellness_env = st.columns(2)
        with trend_cols_wellness_env[0]:
            avg_steps_trend_dist = get_trend_data(filtered_health_records_for_trends, 'avg_daily_steps', period='W', agg_func='mean')
            if not avg_steps_trend_dist.empty: st.plotly_chart(plot_annotated_line_chart(avg_steps_trend_dist, "Weekly Avg. Patient Steps", y_axis_title="Avg. Steps", target_line=app_config.TARGET_DAILY_STEPS), use_container_width=True)
            else: st.caption("No patient steps trend.")
        with trend_cols_wellness_env[1]:
            if not filtered_iot_records_for_trends.empty:
                district_avg_co2_trend = get_trend_data(filtered_iot_records_for_trends, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not district_avg_co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(district_avg_co2_trend, "Daily Avg. CO2 (All Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM), use_container_width=True)
                else: st.caption("No CO2 trend.")
            else: st.caption("No IoT data for environmental trends.")
    else:
        st.info("Select a valid date range or check data source for trend analysis.")

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
            "Population": {"col": "population", "format": "{:,.0f}"},
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
            # Data Table for Comparison
            st.subheader("Zonal Statistics Overview")
            # ... (dataframe styling and display logic as in previous complete file) ...
            display_comp_cols_dist = ['name'] + [d['col'] for d in available_comp_metrics_dist.values() if d['col'] != 'name']; display_comp_cols_dist = list(dict.fromkeys(display_comp_cols_dist))
            comparison_df_display_dist = district_map_and_zonal_stats_gdf[display_comp_cols_dist].copy(); comparison_df_display_dist.rename(columns={'name':'Zone Name'}, inplace=True); comparison_df_display_dist.set_index('Zone Name', inplace=True)
            style_format_dict_dist = {details["col"]: details["format"] for disp_name, details in available_comp_metrics_dist.items() if "format" in details and details["col"] in comparison_df_display_dist.columns}
            highlight_max_cols_dist = [d["col"] for disp_name, d in available_comp_metrics_dist.items() if not d.get("higher_is_worse", False) and d["col"] in comparison_df_display_dist.columns] # Columns where higher is better
            highlight_min_cols_dist = [d["col"] for disp_name, d in available_comp_metrics_dist.items() if d.get("higher_is_worse", True) and d["col"] in comparison_df_display_dist.columns] # Columns where lower is better
            st.dataframe(comparison_df_display_dist.style.format(style_format_dict_dist).background_gradient(cmap='Greens_r', axis=0, subset=pd.IndexSlice[:, highlight_max_cols_dist]).background_gradient(cmap='Reds_r', axis=0, subset=pd.IndexSlice[:, highlight_min_cols_dist]), use_container_width=True, height=min(len(comparison_df_display_dist) * 35 + 40, 450))
            
            st.subheader("Visual Comparison by Metric")
            selected_comp_metric_display_dist = st.selectbox("Select metric for bar chart:", list(available_comp_metrics_dist.keys()), key="district_comp_bar_final_v3") # Unique key
            selected_details_dist = available_comp_metrics_dist.get(selected_comp_metric_display_dist)
            if selected_details_dist:
                selected_col_name_dist = selected_details_dist["col"]
                sort_ascending_bar_dist = selected_details_dist.get("higher_is_worse", True) == False # If higher is worse, sort descending (False)
                bar_df_comp_dist = district_map_and_zonal_stats_gdf[['name', selected_col_name_dist]].copy()
                st.plotly_chart(plot_bar_chart(bar_df_comp_dist, x_col='name', y_col=selected_col_name_dist, title=f"{selected_comp_metric_display_dist} by Zone", x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 100, sort_values_by=selected_col_name_dist, ascending=sort_ascending_bar_dist, text_auto = '.2s' if "population" in selected_col_name_dist.lower() else True), use_container_width=True)
        else: st.info("No metrics available for Zonal Comparison.") # pragma: no cover
    else: st.info("No zonal data available for comparison. Check data loading and enrichment.") # pragma: no cover


with tab_interventions:
    st.header("🛠️ Intervention Planning Insights")
    if not district_map_and_zonal_stats_gdf.empty:
        st.markdown("Identify zones based on combined risk factors. Data reflects overall aggregates.")
        criteria_options_int = {
            f"High Avg. Risk Score (>= {app_config.RISK_THRESHOLDS['district_zone_high_risk']})": "high_risk",
            "Low Facility Coverage (< 50%)": "low_coverage",
            "High Prevalence (Key Inf., Top 25%)": "high_prevalence",
            f"High TB Burden (> 5 active cases)": "high_tb_burden", # Example threshold
            f"High Malaria Burden (> 10 active cases)": "high_malaria_burden", # Example threshold
            f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)": "high_clinic_co2"
        }
        # Filter criteria options based on available columns
        available_criteria_int = {}
        if 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"High Avg. Risk Score (>= {app_config.RISK_THRESHOLDS['district_zone_high_risk']})"] = "high_risk"
        if 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns: available_criteria_int["Low Facility Coverage (< 50%)"] = "low_coverage"
        if 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns: available_criteria_int["High Prevalence (Key Inf., Top 25%)"] = "high_prevalence"
        if 'active_tb_cases' in district_map_and_zonal_stats_gdf.columns: available_criteria_int["High TB Burden (> 5 active cases)"] = "high_tb_burden"
        if 'active_malaria_cases' in district_map_and_zonal_stats_gdf.columns: available_criteria_int["High Malaria Burden (> 10 active cases)"] = "high_malaria_burden"
        if 'zone_avg_co2' in district_map_and_zonal_stats_gdf.columns: available_criteria_int[f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)"] = "high_clinic_co2"
        
        if not available_criteria_int: # pragma: no cover
            st.warning("No criteria can be applied due to missing data columns for intervention planning.")
        else:
            selected_criteria_keys_int = st.multiselect(
                "Filter zones by (meets ANY selected criteria):",
                options=list(available_criteria_int.keys()),
                default=list(available_criteria_int.keys())[0:1] if available_criteria_int else [], # Default to first if available
                key="intervention_criteria_final_v2"
            )
            priority_masks_int = []
            if not selected_criteria_keys_int: st.info("Select criteria to identify priority zones.")
            else:
                for key_int in selected_criteria_keys_int:
                    criterion_int = available_criteria_int[key_int] # Use from filtered list
                    if criterion_int == "high_risk": priority_masks_int.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
                    elif criterion_int == "low_coverage": priority_masks_int.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < 50)
                    elif criterion_int == "high_prevalence" and district_map_and_zonal_stats_gdf['prevalence_per_1000'].notna().any():
                        prev_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
                        if pd.notna(prev_q75): priority_masks_int.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prev_q75)
                    elif criterion_int == "high_tb_burden": priority_masks_int.append(district_map_and_zonal_stats_gdf['active_tb_cases'] > 5)
                    elif criterion_int == "high_malaria_burden": priority_masks_int.append(district_map_and_zonal_stats_gdf['active_malaria_cases'] > 10)
                    elif criterion_int == "high_clinic_co2": priority_masks_int.append(district_map_and_zonal_stats_gdf['zone_avg_co2'] > app_config.CO2_LEVEL_IDEAL_PPM)
                
                if priority_masks_int:
                    try:
                        aligned_criteria_int = [s.reindex(district_map_and_zonal_stats_gdf.index).fillna(False) for s in priority_masks_int if isinstance(s, pd.Series)]
                        if aligned_criteria_int: final_priority_mask_int = pd.concat(aligned_criteria_int, axis=1).any(axis=1)
                        else: final_priority_mask_int = pd.Series([False]*len(district_map_and_zonal_stats_gdf), index=district_map_and_zonal_stats_gdf.index)
                        priority_zones_df_int = district_map_and_zonal_stats_gdf[final_priority_mask_int]
                    except Exception as e_mask_int: logger.error(f"Error applying intervention criteria: {e_mask_int}", exc_info=True); priority_zones_df_int = pd.DataFrame() # pragma: no cover
                    
                    if not priority_zones_df_int.empty:
                        st.markdown("###### Zones Identified for Potential Intervention (Overall Aggregates):")
                        intervention_cols_show = ['name', 'population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2']
                        actual_int_cols = [col for col in intervention_cols_show if col in priority_zones_df_int.columns]
                        sort_by_cols_int = [col for col in ['avg_risk_score', 'active_tb_cases', 'zone_avg_co2'] if col in actual_int_cols] 
                        ascending_int = [False, False, False] # Higher risk, more TB, higher CO2 = higher priority
                        priority_zones_df_int_sorted = priority_zones_df_int.sort_values(by=sort_by_cols_int, ascending=ascending_int) if sort_by_cols_int else priority_zones_df_int
                        
                        format_intervention_table = {col: "{:,.0f}" for col in ['population', 'active_tb_cases', 'active_malaria_cases', 'zone_avg_co2']}
                        format_intervention_table.update({col: "{:.1f}" for col in ['avg_risk_score', 'prevalence_per_1000']})
                        format_intervention_table['facility_coverage_score'] = "{:.1f}%"
                        st.dataframe(priority_zones_df_int_sorted[actual_int_cols], use_container_width=True, 
                                     column_config={k:st.column_config.NumberColumn(format=v.replace('%','%%')) for k,v in format_intervention_table.items() if k in actual_int_cols} )
                    else: st.success("✅ No zones meet selected high-priority criteria.")
                elif selected_criteria_keys_int : st.warning("Could not apply selected criteria effectively. Check data columns.") # pragma: no cover
    else: st.info("No zonal data for intervention insights. Check data loading.") # pragma: no cover
