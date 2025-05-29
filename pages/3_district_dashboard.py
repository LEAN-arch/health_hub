# health_hub/pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import logging
import numpy as np # Added for potential use in calculations
from config import app_config # Centralized configuration
from utils.core_data_processing import (
    load_health_records, load_zone_data, load_iot_clinic_environment_data, # Robust loaders
    enrich_zone_geodata_with_health_aggregates, # Core enrichment function
    get_district_summary_kpis, get_trend_data, hash_geodataframe # Analytics and helpers
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_layered_choropleth_map, plot_annotated_line_chart, # UI helpers
    plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Logger for this page

@st.cache_resource # Cache CSS loading
def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("District Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"District Dashboard: CSS file not found at {app_config.STYLE_CSS}. Default Streamlit styles will be used.")
load_css()

# --- Data Loading (Cached for performance) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None,
    gpd.GeoDataFrame: hash_geodataframe # Use custom hasher for GeoDataFrames
})
def get_district_dashboard_data():
    logger.info("District Dashboard: Attempting to load all necessary data sources...")
    health_df = load_health_records()
    zone_gdf_base = load_zone_data() # This should return a GeoDataFrame with CRS
    iot_df = load_iot_clinic_environment_data()

    # Critical check: Zone geographic data is essential for this dashboard
    if zone_gdf_base is None or zone_gdf_base.empty or 'geometry' not in zone_gdf_base.columns:
        logger.error("CRITICAL - District Dashboard: Base zone geographic data (zone_gdf_base) could not be loaded, is empty, or lacks geometry. Map and most zonal analyses will be unavailable.")
        # Do not call st.error here as this is a data loading function. Error will be shown in main page rendering.
        # Return minimal structures to prevent immediate app crash if this data is consumed before main page check.
        return (health_df if health_df is not None else pd.DataFrame(),
                gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs=app_config.DEFAULT_CRS),
                iot_df if iot_df is not None else pd.DataFrame())

    logger.info("District Dashboard: Enriching zone geographic data with health and IoT aggregates...")
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_gdf_base, health_df, iot_df
    )
    
    if enriched_zone_gdf.empty and not zone_gdf_base.empty:
        logger.warning("District Dashboard: Enrichment of zone GDF resulted in an empty GDF. This may indicate no matching health/IoT data for zones. Falling back to base zone data to show map shapes if possible.")
        # Fallback: Ensure some expected columns exist on base_gdf if enrichment fails, so map plot doesn't entirely break
        for col_map_default in ['avg_risk_score', 'population', 'name']:
             if col_map_default not in zone_gdf_base.columns:
                zone_gdf_base[col_map_default] = 0.0 if col_map_default != 'name' else "Unknown Zone"
        return health_df, zone_gdf_base, iot_df # Return base GDF instead of empty enriched one

    logger.info("District Dashboard: Data loading and enrichment process complete.")
    return health_df, enriched_zone_gdf, iot_df

# Load all data for the dashboard
health_records_district_main, district_gdf_main_enriched, iot_records_district_main = get_district_dashboard_data()

# --- Main Page Structure ---
# Handle critical data failure for GDF at the page level before attempting to render anything major
if district_gdf_main_enriched is None or district_gdf_main_enriched.empty or 'geometry' not in district_gdf_main_enriched.columns or district_gdf_main_enriched.geometry.is_empty.all():
    st.error("🚨 **CRITICAL Error:** Geographic zone data is missing, invalid, or empty. The District Dashboard cannot be rendered. Please verify 'zone_geometries.geojson' and its link with 'zone_attributes.csv'.")
    logger.critical("District Dashboard HALTED: district_gdf_main_enriched is unusable.")
    st.stop() # Halt execution

st.title("🗺️ District Health Officer (DHO) Dashboard")
st.markdown("**Strategic Population Health Insights, Resource Allocation, and Environmental Well-being Monitoring**")
st.markdown("---")

# --- Sidebar Filters for Trend Date Range ---
st.sidebar.header("🗓️ District Filters")

all_dates_for_district_trends_raw = []
if health_records_district_main is not None and 'date' in health_records_district_main and health_records_district_main['date'].notna().any():
    valid_health_dates_dist = pd.to_datetime(health_records_district_main['date'], errors='coerce').dropna()
    if not valid_health_dates_dist.empty: all_dates_for_district_trends_raw.extend(valid_health_dates_dist)
if iot_records_district_main is not None and 'timestamp' in iot_records_district_main and iot_records_district_main['timestamp'].notna().any():
    valid_iot_dates_dist = pd.to_datetime(iot_records_district_main['timestamp'], errors='coerce').dropna()
    if not valid_iot_dates_dist.empty: all_dates_for_district_trends_raw.extend(valid_iot_dates_dist)

if not all_dates_for_district_trends_raw:
    min_date_for_trends_dist = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 6)
    max_date_for_trends_dist = pd.Timestamp('today').date()
    logger.warning("District Dashboard: No valid dates in any dataset for trend filter. Using wide fallback.")
else:
    all_dates_series_dist = pd.Series(all_dates_for_district_trends_raw).dropna()
    if all_dates_series_dist.empty:
        min_date_for_trends_dist = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 6)
        max_date_for_trends_dist = pd.Timestamp('today').date()
        logger.warning("District Dashboard: Date series for trends became empty after processing. Using wide fallback.")
    else:
        min_date_for_trends_dist = all_dates_series_dist.min().date()
        max_date_for_trends_dist = all_dates_series_dist.max().date()

default_start_date_dist_trends = max_date_for_trends_dist - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_date_dist_trends < min_date_for_trends_dist : default_start_date_dist_trends = min_date_for_trends_dist
if default_start_date_dist_trends > max_date_for_trends_dist : default_start_date_dist_trends = max_date_for_trends_dist

selected_start_date_dist_trends, selected_end_date_dist_trends = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:", value=[default_start_date_dist_trends, max_date_for_trends_dist],
    min_value=min_date_for_trends_dist, max_value=max_date_for_trends_dist, key="district_trends_date_selector_final",
    help="This date range applies to time-series trend charts for health and environmental data."
)

# Filter health_records and IoT records for the selected trend period
# Ensure 'date_obj' exists before filtering
filtered_health_for_trends = pd.DataFrame()
if health_records_district_main is not None and not health_records_district_main.empty and 'date' in health_records_district_main.columns:
    if not pd.api.types.is_datetime64_ns_dtype(health_records_district_main['date']): health_records_district_main['date'] = pd.to_datetime(health_records_district_main['date'], errors='coerce')
    if 'date_obj' not in health_records_district_main.columns: health_records_district_main['date_obj'] = health_records_district_main['date'].dt.date
    
    mask_health_trends = (health_records_district_main['date_obj'] >= selected_start_date_dist_trends) & \
                         (health_records_district_main['date_obj'] <= selected_end_date_dist_trends)
    filtered_health_for_trends = health_records_district_main[mask_health_trends].copy()

filtered_iot_for_trends = pd.DataFrame()
if iot_records_district_main is not None and not iot_records_district_main.empty and 'timestamp' in iot_records_district_main.columns:
    if not pd.api.types.is_datetime64_ns_dtype(iot_records_district_main['timestamp']): iot_records_district_main['timestamp'] = pd.to_datetime(iot_records_district_main['timestamp'], errors='coerce')
    if 'date_obj' not in iot_records_district_main.columns: iot_records_district_main['date_obj'] = iot_records_district_main['timestamp'].dt.date

    mask_iot_trends = (iot_records_district_main['date_obj'] >= selected_start_date_dist_trends) & \
                      (iot_records_district_main['date_obj'] <= selected_end_date_dist_trends)
    filtered_iot_for_trends = iot_records_district_main[mask_iot_trends].copy()


# --- KPIs Section (based on overall aggregates from `district_gdf_main_enriched`) ---
st.subheader("District-Wide Key Performance Indicators (Aggregated Zonal Data)")
district_overall_kpis = get_district_summary_kpis(district_gdf_main_enriched) # Handles empty/None GDF

kpi_cols_row1_dist = st.columns(4)
with kpi_cols_row1_dist[0]:
    avg_pop_risk_val = district_overall_kpis.get('avg_population_risk', 0.0)
    render_kpi_card("Avg. Population Risk", f"{avg_pop_risk_val:.1f}", "🎯",
                    status="High" if avg_pop_risk_val > app_config.RISK_THRESHOLDS['high'] else "Moderate" if avg_pop_risk_val > app_config.RISK_THRESHOLDS['moderate'] else "Low",
                    help_text="Population-weighted average AI risk score across all zones.")
with kpi_cols_row1_dist[1]:
    facility_coverage_val = district_overall_kpis.get('overall_facility_coverage', 0.0)
    render_kpi_card("Facility Coverage", f"{facility_coverage_val:.1f}%", "🏥",
                    status="Bad Low" if facility_coverage_val < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD else "Moderate" if facility_coverage_val < 80 else "Good High",
                    help_text="Population-weighted score reflecting access and capacity of health facilities.")
with kpi_cols_row1_dist[2]:
    high_risk_zones_num = district_overall_kpis.get('zones_high_risk_count', 0)
    total_zones_val = len(district_gdf_main_enriched) if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 1
    perc_high_risk_zones = (high_risk_zones_num / total_zones_val) * 100 if total_zones_val > 0 else 0
    render_kpi_card("High-Risk Zones", f"{high_risk_zones_num} ({perc_high_risk_zones:.0f}%)", "⚠️",
                    status="High" if perc_high_risk_zones > 25 else "Moderate" if high_risk_zones_num > 0 else "Low",
                    help_text=f"Number (and %) of zones with average risk score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
with kpi_cols_row1_dist[3]:
    district_prevalence_val = district_overall_kpis.get('key_infection_prevalence_district_per_1000', 0.0)
    render_kpi_card("Overall Prevalence", f"{district_prevalence_val:.1f} /1k Pop", "📈",
                    status="High" if district_prevalence_val > 50 else ("Moderate" if district_prevalence_val > 10 else "Low"),
                    help_text="Combined prevalence of key infectious diseases per 1,000 population in the district.")

st.markdown("##### Key Disease Burdens & District Wellness / Environment")
kpi_cols_row2_dist = st.columns(4)
with kpi_cols_row2_dist[0]:
    tb_total_burden = district_overall_kpis.get('district_tb_burden_total', 0)
    render_kpi_card("Active TB Cases", str(tb_total_burden), "<img src='https://www.svgrepo.com/show/309948/lungs.svg' width='30' alt='TB'>", icon_is_html=True,
                    status="High" if tb_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None else 50) else "Moderate", # Example: threshold based on num_zones * per_zone_thresh
                    help_text="Total active TB cases identified across the district (latest aggregates).")
with kpi_cols_row2_dist[1]:
    malaria_total_burden = district_overall_kpis.get('district_malaria_burden_total',0)
    render_kpi_card("Active Malaria Cases", str(malaria_total_burden), "<img src='https://www.svgrepo.com/show/491020/mosquito.svg' width='30' alt='Malaria'>", icon_is_html=True,
                    status="High" if malaria_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None else 100) else "Moderate",
                    help_text="Total active Malaria cases identified across the district (latest aggregates).")
with kpi_cols_row2_dist[2]:
    avg_steps_district = district_overall_kpis.get('population_weighted_avg_steps', 0.0)
    render_kpi_card("Avg. Patient Steps", f"{avg_steps_district:,.0f}", "👣",
                    status="Bad Low" if avg_steps_district < (app_config.TARGET_DAILY_STEPS * 0.7) else "Moderate" if avg_steps_district < app_config.TARGET_DAILY_STEPS else "Good High",
                    help_text=f"Population-weighted average daily steps. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
with kpi_cols_row2_dist[3]:
    avg_co2_district_val = district_overall_kpis.get('avg_clinic_co2_district',0.0)
    render_kpi_card("Avg. Clinic CO2", f"{avg_co2_district_val:.0f} ppm", "💨",
                    status="High" if avg_co2_district_val > app_config.CO2_LEVEL_ALERT_PPM else "Moderate" if avg_co2_district_val > app_config.CO2_LEVEL_IDEAL_PPM else "Low",
                    help_text="District average of zonal mean CO2 levels in clinics (unweighted average of zonal means).")
st.markdown("---")

# --- Choropleth Map Section ---
st.subheader("🗺️ Interactive Health & Environment Map of the District")
map_metric_options_config_dist = {
    "Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale": "Reds_r"}, # Higher is worse
    "Total Key Infections": {"col": "total_active_key_infections", "colorscale": "OrRd_r"},
    "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r"},
    "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale": "Greens"}, # Higher is better
    "Active TB Cases": {"col": "active_tb_cases", "colorscale": "Purples_r"},
    "Active Malaria Cases": {"col": "active_malaria_cases", "colorscale": "Oranges_r"},
    "Avg. Patient Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "Cividis"}, # Higher is better
    "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "colorscale": " agencia"}, # Lower is better, (reds are bad)
    "Number of Clinics": {"col": "num_clinics", "colorscale": "Blues"}, # More is generally better
    "Socio-Economic Index": {"col": "socio_economic_index", "colorscale": "Tealgrn"}
}
if 'population' in district_gdf_main_enriched.columns:
    try: 
        gdf_proj_area = district_gdf_main_enriched.to_crs(district_gdf_main_enriched.estimate_utm_crs()) if district_gdf_main_enriched.crs else None
        if gdf_proj_area is not None: district_gdf_main_enriched['area_sqkm'] = gdf_proj_area.geometry.area / 1_000_000
        else: district_gdf_main_enriched['area_sqkm'] = np.nan
        district_gdf_main_enriched['population_density'] = district_gdf_main_enriched.apply(lambda r: r['population'] / r['area_sqkm'] if pd.notna(r['area_sqkm']) and r['area_sqkm']>0 else 0, axis=1)
        map_metric_options_config_dist["Population Density (Pop/SqKm)"] = {"col": "population_density", "colorscale": "Plasma_r"}
    except Exception as e_map_area: logger.warning(f"Map: Could not calculate area/pop density for map metric options: {e_map_area}")

available_map_metrics_for_select = {
    disp_name: details for disp_name, details in map_metric_options_config_dist.items()
    if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
}

if available_map_metrics_for_select:
    selected_map_metric_display_name = st.selectbox(
        "Select Metric to Visualize on Map:", list(available_map_metrics_for_select.keys()),
        key="district_map_metric_selector_final",
        help="Choose a health, resource, or environmental metric for spatial visualization."
    )
    selected_map_metric_config = available_map_metrics_for_select.get(selected_map_metric_display_name)

    if selected_map_metric_config:
        map_val_col = selected_map_metric_config["col"]
        map_colorscale = selected_map_metric_config["colorscale"]
        
        hover_cols_for_map = ['name', 'population', map_val_col] # Base hover info
        if 'num_clinics' in district_gdf_main_enriched.columns and map_val_col != 'num_clinics': hover_cols_for_map.append('num_clinics')
        if 'facility_coverage_score' in district_gdf_main_enriched.columns and map_val_col != 'facility_coverage_score': hover_cols_for_map.append('facility_coverage_score')
        final_hover_cols_map = list(dict.fromkeys([col for col in hover_cols_for_map if col in district_gdf_main_enriched.columns]))

        map_figure = plot_layered_choropleth_map(
            gdf=district_gdf_main_enriched, value_col=map_val_col, title=f"District Map: {selected_map_metric_display_name}",
            id_col='zone_id', featureidkey_prefix='properties',
            color_continuous_scale=map_colorscale, hover_cols=final_hover_cols_map,
            height=app_config.MAP_PLOT_HEIGHT, mapbox_style=app_config.MAPBOX_STYLE
        )
        st.plotly_chart(map_figure, use_container_width=True)
    else: st.info("Please select a metric to display on the map.")
else: st.warning("No metrics with valid data are currently available for map visualization in the enriched GeoDataFrame.")
st.markdown("---")

# --- Tabs for Detailed Analysis ---
tab_dist_trends, tab_dist_comparison, tab_dist_interventions = st.tabs([
    "📈 District-Wide Trends", "📊 Zonal Comparative Analysis", "🎯 Intervention Planning Insights"
])

with tab_dist_trends:
    st.header("📈 District-Wide Health & Environmental Trends")
    if not filtered_health_for_trends.empty or not filtered_iot_for_trends.empty:
        st.markdown(f"Displaying trends from **{selected_start_date_dist_trends.strftime('%d %b %Y')}** to **{selected_end_date_dist_trends.strftime('%d %b %Y')}**.")
        st.subheader("Key Disease Incidence Trends (New Cases per Week)")
        cols_disease_trends = st.columns(2)
        with cols_disease_trends[0]:
            if not filtered_health_for_trends.empty and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
                tb_trends_src = filtered_health_for_trends[filtered_health_for_trends['condition'] == 'TB']
                weekly_tb_trend = get_trend_data(tb_trends_src, 'patient_id', date_col='date', period='W', agg_func='nunique')
                if not weekly_tb_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_tb_trend, "Weekly New TB Patients", y_axis_title="New TB Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No TB trend data for this period.")
            else: st.caption("TB trend data cannot be generated (check data columns or period).")
        with cols_disease_trends[1]:
            if not filtered_health_for_trends.empty and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
                malaria_trends_src = filtered_health_for_trends[filtered_health_for_trends['condition'] == 'Malaria']
                weekly_malaria_trend = get_trend_data(malaria_trends_src, 'patient_id', date_col='date', period='W', agg_func='nunique')
                if not weekly_malaria_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_malaria_trend, "Weekly New Malaria Patients", y_axis_title="New Malaria Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No Malaria trend data for this period.")
            else: st.caption("Malaria trend data unavailable.")

        st.subheader("Population Wellness & Environmental Trends")
        cols_wellness_env = st.columns(2)
        with cols_wellness_env[0]:
            if not filtered_health_for_trends.empty and 'avg_daily_steps' in filtered_health_for_trends.columns:
                steps_trends_dist = get_trend_data(filtered_health_for_trends, 'avg_daily_steps', date_col='date', period='W', agg_func='mean')
                if not steps_trends_dist.empty: st.plotly_chart(plot_annotated_line_chart(steps_trends_dist, "Weekly Avg. Patient Steps", y_axis_title="Avg. Steps", target_line=app_config.TARGET_DAILY_STEPS, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No steps trend data for this period.")
            else: st.caption("Avg. daily steps data missing for trends.")
        with cols_wellness_env[1]:
            if not filtered_iot_for_trends.empty and 'avg_co2_ppm' in filtered_iot_for_trends.columns:
                co2_trends_dist_iot = get_trend_data(filtered_iot_for_trends, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not co2_trends_dist_iot.empty: st.plotly_chart(plot_annotated_line_chart(co2_trends_dist_iot, "Daily Avg. CO2 (All Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No CO2 trend data from clinics for this period.")
            else: st.caption("Clinic CO2 data missing for trends.")
    else: st.info("No trend data available for the selected period. Please adjust the date range or verify data sources.")


with tab_dist_comparison:
    st.header("📊 Zonal Comparative Analysis")
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched:
        st.markdown("Compare zones using aggregated health, resource, environmental, and socio-economic metrics. Data is based on latest aggregations.")
        # Reuse map_metric_options_config_dist but filter for table needs
        comp_table_metrics_dict = {
            name: details for name, details in map_metric_options_config_dist.items()
            if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
        }
        
        if comp_table_metrics_dict:
            st.subheader("Zonal Statistics Table")
            table_cols_to_display = ['name'] + [d['col'] for d in comp_table_metrics_dict.values()]
            df_for_comp_table = district_gdf_main_enriched[[col for col in table_cols_to_display if col in district_gdf_main_enriched.columns]].copy()
            df_for_comp_table.rename(columns={'name':'Zone'}, inplace=True)
            if 'Zone' in df_for_comp_table.columns: df_for_comp_table.set_index('Zone', inplace=True)

            styled_comp_table = df_for_comp_table.style
            for metric_name, metric_details in comp_table_metrics_dict.items():
                col_name_style = metric_details['col']
                if col_name_style in df_for_comp_table.columns:
                    styled_comp_table = styled_comp_table.format({col_name_style: metric_details.get("format", "{:.1f}")})
                    # Determine if higher values are bad (use Reds_r) or good (use Greens)
                    cmap_gradient = 'Reds_r' if "_r" in metric_details.get("colorscale", "Reds_r") else 'Greens' # If colorscale ends with _r, it implies higher is worse
                    if "Greens" in metric_details.get("colorscale",""): cmap_gradient = "Greens"

                    try: styled_comp_table = styled_comp_table.background_gradient(subset=[col_name_style], cmap=cmap_gradient, axis=0)
                    except: pass # Ignore minor styling errors

            st.dataframe(styled_comp_table, use_container_width=True, height=min(len(df_for_comp_table) * 45 + 60, 600))

            st.subheader("Visual Comparison Chart")
            selected_bar_metric_name_dist_comp = st.selectbox(
                "Select Metric for Bar Chart Comparison:", list(comp_table_metrics_dict.keys()),
                key="district_comparison_barchart_final_selector"
            )
            selected_bar_details_dist_comp = comp_table_metrics_dict.get(selected_bar_metric_name_dist_comp)
            if selected_bar_details_dist_comp:
                bar_col_for_comp = selected_bar_details_dist_comp["col"]
                text_format_bar_comp = selected_bar_details_dist_comp.get("format", "{:.1f}").replace('{','').replace('}','').split(':')[-1]
                # Higher is worse typically means we sort descending to see worst first. Colorscale suffix _r helps.
                sort_asc_bar = "_r" not in selected_bar_details_dist_comp.get("colorscale", "") 

                st.plotly_chart(plot_bar_chart(
                    district_gdf_main_enriched, x_col='name', y_col=bar_col_for_comp,
                    title=f"{selected_bar_metric_name_dist_comp} by Zone",
                    x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 150,
                    sort_values_by=bar_col_for_comp, ascending=sort_asc_bar,
                    text_auto=True, text_format=text_format_bar_comp
                ), use_container_width=True)
        else: st.info("No metrics available for Zonal Comparison table/chart.")
    else: st.info("Zonal comparison requires enriched geographic data. Please check data loading.")


with tab_dist_interventions:
    st.header("🎯 Intervention Planning Insights")
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched:
        st.markdown("Identify zones for targeted interventions based on customizable criteria related to health risks, disease burdens, resource accessibility, and environmental factors.")
        # Define criteria using lambdas for flexibility and to access GDF columns robustly
        # These functions expect the GDF as input and return a boolean Series
        criteria_lambdas_intervention = {
            f"High Avg. Risk (≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']})":
                lambda df: df.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.RISK_THRESHOLDS['district_zone_high_risk'],
            f"Low Facility Coverage (< {app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD}%)":
                lambda df: df.get('facility_coverage_score', pd.Series(dtype=float)) < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD,
            f"High Key Inf. Prevalence (Top {100-app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE*100:.0f}%)":
                lambda df: df.get('prevalence_per_1000', pd.Series(dtype=float)) >= df['prevalence_per_1000'].quantile(app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE) if 'prevalence_per_1000' in df and df['prevalence_per_1000'].notna().any() else pd.Series([False]*len(df), index=df.index),
            f"High TB Burden (Abs. > {app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD})":
                lambda df: df.get('active_tb_cases', pd.Series(dtype=float)) > app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD,
            f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)":
                lambda df: df.get('zone_avg_co2', pd.Series(dtype=float)) > app_config.CO2_LEVEL_IDEAL_PPM
        }
        # Dynamically create options based on available data in GDF for the lambdas
        available_criteria_for_intervention = {}
        for name, func_lambda in criteria_lambdas_intervention.items():
            # Test if the lambda can run without error (e.g., if primary col it uses exists)
            try: func_lambda(district_gdf_main_enriched.head(1)) # Test with a small slice
            except (KeyError, AttributeError): continue # Skip if underlying column is missing for this criterion
            available_criteria_for_intervention[name] = func_lambda


        if not available_criteria_for_intervention:
            st.warning("Intervention criteria cannot be applied; relevant data columns may be missing from the enriched zone data.")
        else:
            selected_criteria_display_names = st.multiselect(
                "Select Criteria to Identify Priority Zones (Zones meeting ANY selected criteria will be shown):",
                options=list(available_criteria_for_intervention.keys()),
                default=list(available_criteria_for_intervention.keys())[0:min(2, len(available_criteria_for_intervention))] if available_criteria_for_intervention else [], # Sensible default: first two criteria
                key="district_intervention_criteria_selector_final",
                help="Choose one or more criteria. Zones satisfying any of these will be listed."
            )

            if not selected_criteria_display_names:
                st.info("Please select at least one criterion to identify priority zones for intervention.")
            else:
                final_intervention_mask = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
                for crit_name_selected in selected_criteria_display_names:
                    crit_func_selected = available_criteria_for_intervention[crit_name_selected]
                    try:
                        current_crit_mask = crit_func_selected(district_gdf_main_enriched)
                        if isinstance(current_crit_mask, pd.Series) and current_crit_mask.dtype == 'bool':
                             final_intervention_mask = final_intervention_mask | current_crit_mask.fillna(False)
                        else: logger.warning(f"Intervention criterion '{crit_name_selected}' did not return a valid boolean Series. It will be ignored.")
                    except Exception as e_crit_apply:
                        logger.error(f"Error applying intervention criterion '{crit_name_selected}': {e_crit_apply}", exc_info=True)
                        st.warning(f"Could not apply criterion: {crit_name_selected}. Error: {e_crit_apply}")
                
                priority_zones_df_interv = district_gdf_main_enriched[final_intervention_mask].copy()

                if not priority_zones_df_interv.empty:
                    st.markdown(f"###### Identified **{len(priority_zones_df_interv)}** Zone(s) Meeting Selected Intervention Criteria:")
                    
                    cols_intervention_table = ['name', 'population', 'avg_risk_score', 'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2']
                    actual_cols_interv_table = [col for col in cols_intervention_table if col in priority_zones_df_interv.columns]
                    
                    # Sort by highest risk, then highest prevalence, then lowest coverage to prioritize
                    sort_by_list_interv = []; sort_asc_list_interv = []
                    if 'avg_risk_score' in actual_cols_interv_table: sort_by_list_interv.append('avg_risk_score'); sort_asc_list_interv.append(False)
                    if 'prevalence_per_1000' in actual_cols_interv_table: sort_by_list_interv.append('prevalence_per_1000'); sort_asc_list_interv.append(False)
                    if 'facility_coverage_score' in actual_cols_interv_table: sort_by_list_interv.append('facility_coverage_score'); sort_asc_list_interv.append(True)

                    interv_df_display_sorted = priority_zones_df_interv.sort_values(by=sort_by_list_interv, ascending=sort_asc_list_interv) if sort_by_list_interv else priority_zones_df_interv
                    
                    st.dataframe(
                        interv_df_display_sorted[actual_cols_interv_table], use_container_width=True, hide_index=True,
                        column_config={
                            "name": st.column_config.TextColumn("Zone Name", help="Administrative zone name."),
                            "population": st.column_config.NumberColumn("Population", format="%,.0f"),
                            "avg_risk_score": st.column_config.ProgressColumn("Avg. Risk Score", format="%.1f", min_value=0, max_value=100, help="Average AI-calculated population risk."),
                            "total_active_key_infections": st.column_config.NumberColumn("Total Key Infections", format="%.0f"),
                            "prevalence_per_1000": st.column_config.NumberColumn("Prevalence (/1k Pop.)", format="%.1f"),
                            "facility_coverage_score": st.column_config.NumberColumn("Facility Coverage (%)", format="%.1f%%", help="Composite score of facility access & capacity."),
                            "zone_avg_co2": st.column_config.NumberColumn("Avg. Clinic CO2 (ppm)", format="%.0f ppm")
                        }
                    )
                else:
                    st.success("✅ No zones currently meet the selected criteria for intervention based on the available aggregated data.")
    else:
        st.info("Intervention planning insights require successfully loaded and enriched geographic zone data. Please check data sources and processing.")
