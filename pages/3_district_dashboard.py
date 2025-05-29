# health_hub/pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import logging
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
        st.error("🚨 **CRITICAL Error:** Geographic zone data is missing or invalid. The District Dashboard cannot be fully rendered. Please check 'zone_geometries.geojson' and 'zone_attributes.csv'.")
        # Return minimal structures to prevent complete app crash, but functionality will be severely limited.
        return (health_df if health_df is not None else pd.DataFrame(),
                gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs=app_config.DEFAULT_CRS), # Empty GDF with schema
                iot_df if iot_df is not None else pd.DataFrame())

    # Enrich zone data with health and IoT aggregates. This function should handle empty health_df or iot_df.
    logger.info("District Dashboard: Enriching zone geographic data with health and IoT aggregates...")
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_gdf_base, # Must be a valid GeoDataFrame
        health_df,     # Can be an empty DataFrame
        iot_df         # Can be an empty DataFrame
    )
    
    if enriched_zone_gdf.empty and not zone_gdf_base.empty:
        logger.warning("District Dashboard: Enrichment of zone GDF resulted in an empty GDF. This might happen if health/IoT data couldn't be aggregated per zone. Falling back to base zone data for map shapes if possible.")
        # To at least show map shapes, fallback to base_gdf but ensure it has some expected cols for plot functions.
        # This helps if, e.g., only attributes are loaded but no health data to aggregate.
        for col_check in ['avg_risk_score', 'population', 'name']: # Example default cols map might use
             if col_check not in zone_gdf_base.columns: zone_gdf_base[col_check] = 0 if col_check != 'name' else "Unknown Zone"
        enriched_zone_gdf = zone_gdf_base

    logger.info("District Dashboard: Data loading and enrichment process complete.")
    return health_df, enriched_zone_gdf, iot_df

# Load all data for the dashboard
health_records_district_main, district_gdf_main_enriched, iot_records_district_main = get_district_dashboard_data()

# --- Main Page Structure ---
st.title("🗺️ District Health Officer (DHO) Dashboard")
st.markdown("**Strategic Population Health Insights, Resource Allocation, and Environmental Well-being Monitoring**")
st.markdown("---") # Visual separator

# --- Sidebar Filters for Trend Date Range ---
st.sidebar.header("🗓️ District Filters")

# Determine overall available date range for trend filters from all datasets
all_dates_for_district_trends = []
if health_records_district_main is not None and 'date' in health_records_district_main and health_records_district_main['date'].notna().any():
    all_dates_for_district_trends.extend(health_records_district_main['date'])
if iot_records_district_main is not None and 'timestamp' in iot_records_district_main and iot_records_district_main['timestamp'].notna().any():
    all_dates_for_district_trends.extend(iot_records_district_main['timestamp'])

if not all_dates_for_district_trends: # Fallback if no date data found in any source
    min_date_for_trends = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 6) # Wider fallback
    max_date_for_trends = pd.Timestamp('today').date()
    logger.warning("District Dashboard: No date information found in any loaded datasets for trend filter range.")
else:
    min_date_for_trends = pd.Series(all_dates_for_district_trends).min().date()
    max_date_for_trends = pd.Series(all_dates_for_district_trends).max().date()

default_start_date_trends = max_date_for_trends - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_date_trends < min_date_for_trends : default_start_date_trends = min_date_for_trends


selected_start_date_dist_trends, selected_end_date_dist_trends = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:", value=[default_start_date_trends, max_date_for_trends],
    min_value=min_date_for_trends, max_value=max_date_for_trends, key="district_trends_date_selector",
    help="This date range applies to time-series trend charts for health and environmental data."
)

# Filter health_records and IoT records for the selected trend period
filtered_health_for_trends = pd.DataFrame()
if health_records_district_main is not None and not health_records_district_main.empty and 'date' in health_records_district_main.columns:
    health_records_district_main['date_obj'] = pd.to_datetime(health_records_district_main['date']).dt.date
    mask_health_trends = (health_records_district_main['date_obj'] >= selected_start_date_dist_trends) & \
                         (health_records_district_main['date_obj'] <= selected_end_date_dist_trends)
    filtered_health_for_trends = health_records_district_main[mask_health_trends].copy()

filtered_iot_for_trends = pd.DataFrame()
if iot_records_district_main is not None and not iot_records_district_main.empty and 'timestamp' in iot_records_district_main.columns:
    iot_records_district_main['date_obj'] = pd.to_datetime(iot_records_district_main['timestamp']).dt.date
    mask_iot_trends = (iot_records_district_main['date_obj'] >= selected_start_date_dist_trends) & \
                      (iot_records_district_main['date_obj'] <= selected_end_date_dist_trends)
    filtered_iot_for_trends = iot_records_district_main[mask_iot_trends].copy()


# --- KPIs Section (based on overall aggregates from `district_gdf_main_enriched`) ---
st.subheader("District-Wide Key Performance Indicators (Overall Aggregates)")
if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty:
    district_overall_kpis = get_district_summary_kpis(district_gdf_main_enriched)
    kpi_cols_row1_dist = st.columns(4) # More KPIs can fit with refined styling
    with kpi_cols_row1_dist[0]:
        avg_pop_risk_val = district_overall_kpis.get('avg_population_risk', 0.0)
        render_kpi_card("Avg. Population Risk", f"{avg_pop_risk_val:.1f}", "🎯",
                        status="High" if avg_pop_risk_val > app_config.RISK_THRESHOLDS['high'] else "Moderate" if avg_pop_risk_val > app_config.RISK_THRESHOLDS['moderate'] else "Low",
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols_row1_dist[1]:
        facility_coverage_val = district_overall_kpis.get('overall_facility_coverage', 0.0)
        render_kpi_card("Facility Coverage", f"{facility_coverage_val:.1f}%", "🏥",
                        status="Low" if facility_coverage_val < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD else "Moderate" if facility_coverage_val < 80 else "High",
                        help_text="Population-weighted score reflecting access and capacity of health facilities.")
    with kpi_cols_row1_dist[2]:
        high_risk_zones_num = district_overall_kpis.get('zones_high_risk_count', 0)
        total_zones_val = len(district_gdf_main_enriched) if district_gdf_main_enriched is not None else 1 # Avoid div by zero
        perc_high_risk_zones = (high_risk_zones_num / total_zones_val) * 100 if total_zones_val > 0 else 0
        render_kpi_card("High-Risk Zones", f"{high_risk_zones_num} ({perc_high_risk_zones:.0f}%)", "⚠️",
                        status="High" if perc_high_risk_zones > 25 else "Moderate" if high_risk_zones_num > 0 else "Low", # Example: alert if >25% zones are high risk
                        help_text=f"Number (and percentage) of zones with average risk score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    with kpi_cols_row1_dist[3]:
        district_prevalence_val = district_overall_kpis.get('key_infection_prevalence_district_per_1000', 0.0)
        render_kpi_card("Overall Prevalence", f"{district_prevalence_val:.1f} /1000", "📈", # Graph icon
                        status="High" if district_prevalence_val > 50 else "Moderate", # Example threshold for overall prevalence
                        help_text="Combined prevalence of key infectious diseases per 1,000 population in the district.")

    st.markdown("##### Key Disease Burdens & District Wellness / Environment")
    kpi_cols_row2_dist = st.columns(4)
    with kpi_cols_row2_dist[0]:
        tb_total_burden = district_overall_kpis.get('district_tb_burden_total', 0)
        render_kpi_card("Active TB Cases", str(tb_total_burden), "<img src='https://www.svgrepo.com/show/309948/lungs.svg' width='30' alt='TB'>", icon_is_html=True,
                        status="High" if tb_total_burden > 50 else "Moderate", # Example: High if >50 active cases district-wide
                        help_text="Total active TB cases identified across the district (latest aggregates).")
    with kpi_cols_row2_dist[1]:
        malaria_total_burden = district_overall_kpis.get('district_malaria_burden_total',0)
        render_kpi_card("Active Malaria Cases", str(malaria_total_burden), "<img src='https://www.svgrepo.com/show/491020/mosquito.svg' width='30' alt='Malaria'>", icon_is_html=True,
                        status="High" if malaria_total_burden > 100 else "Moderate",
                        help_text="Total active Malaria cases identified across the district (latest aggregates).")
    with kpi_cols_row2_dist[2]:
        avg_steps_district = district_overall_kpis.get('population_weighted_avg_steps', 0.0)
        render_kpi_card("Avg. Patient Steps", f"{avg_steps_district:,.0f}", "👣",
                        status="Low" if avg_steps_district < (app_config.TARGET_DAILY_STEPS * 0.7) else "Moderate" if avg_steps_district < app_config.TARGET_DAILY_STEPS else "High",
                        help_text=f"Population-weighted average daily steps. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    with kpi_cols_row2_dist[3]:
        avg_co2_district_val = district_overall_kpis.get('avg_clinic_co2_district',0.0)
        render_kpi_card("Avg. Clinic CO2", f"{avg_co2_district_val:.0f} ppm", "💨",
                        status="High" if avg_co2_district_val > app_config.CO2_LEVEL_ALERT_PPM else "Moderate" if avg_co2_district_val > app_config.CO2_LEVEL_IDEAL_PPM else "Low",
                        help_text="District average of zonal mean CO2 levels in clinics (unweighted average of zonal means).")
else:
    st.warning("District-Wide KPIs cannot be displayed: Enriched zone geographic data is unavailable. Please check data loading and processing steps.")
st.markdown("---") # Visual separator

# --- Choropleth Map Section ---
st.subheader("🗺️ Interactive Health & Environment Map of the District")
# Check if GDF exists, has geometry, and is not empty before attempting to plot
if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns and not district_gdf_main_enriched.geometry.is_empty.all():
    # Define a more comprehensive set of metrics for map visualization
    map_metric_options_config = {
        "Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale": "Reds"},
        "Total Key Infections": {"col": "total_active_key_infections", "colorscale": "OrRd"},
        "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd"},
        "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale": "Greens"}, # Higher is better
        "Active TB Cases": {"col": "active_tb_cases", "colorscale": "Purples"},
        "Active Malaria Cases": {"col": "active_malaria_cases", "colorscale": "Oranges"},
        "HIV Positive Cases (Agg.)": {"col": "hiv_positive_cases", "colorscale": "Magenta"},
        "Avg. Patient Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "Cividis_r"}, # Higher better
        "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "colorscale": " agencia"}, # Lower is better; Viridis_r for bad higher
        "Population Density (Pop/SqKm)": {"col": "population_density", "colorscale": "Plasma"}, # Requires area calculation
        "Number of Clinics": {"col": "num_clinics", "colorscale": "Blues"},
        "Socio-Economic Index": {"col": "socio_economic_index", "colorscale": "Tealgrn"} # Higher is better
    }
    # Calculate population density if area can be determined (requires projected CRS for GDF for area())
    if 'population' in district_gdf_main_enriched.columns:
        try: # Ensure projected CRS for area calculation
            if district_gdf_main_enriched.crs and not district_gdf_main_enriched.crs.is_geographic:
                district_gdf_main_enriched['area_sqkm'] = district_gdf_main_enriched.geometry.area / 1_000_000 # area in sq meters usually
            else: # Attempt to reproject to a suitable UTM for area
                guessed_utm_crs = district_gdf_main_enriched.estimate_utm_crs()
                if guessed_utm_crs:
                    district_gdf_main_enriched['area_sqkm'] = district_gdf_main_enriched.geometry.to_crs(guessed_utm_crs).area / 1_000_000
                else: district_gdf_main_enriched['area_sqkm'] = np.nan
            district_gdf_main_enriched['population_density'] = district_gdf_main_enriched.apply(lambda r: r['population'] / r['area_sqkm'] if pd.notna(r['area_sqkm']) and r['area_sqkm']>0 else 0, axis=1)
        except Exception as e_area:
            logger.warning(f"Could not calculate area/population density for map: {e_area}")
            district_gdf_main_enriched['population_density'] = np.nan # Ensure col exists if calculation failed


    # Filter map options based on available and non-null columns in the GDF
    available_map_options = {
        display_name: details for display_name, details in map_metric_options_config.items()
        if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
    }

    if available_map_options:
        selected_map_metric_name = st.selectbox(
            "Select Metric to Visualize on Map:", list(available_map_options.keys()),
            key="district_interactive_map_metric_selector",
            help="Choose a health, resource, or environmental metric to visualize spatially across the district zones."
        )
        selected_metric_details = available_map_options.get(selected_map_metric_name)

        if selected_metric_details:
            map_value_col = selected_metric_details["col"]
            map_color_scale_theme = selected_metric_details["colorscale"]
            
            # Define common hover columns for richer tooltips
            hover_data_map = ['name', 'population', map_value_col]
            if 'num_clinics' in district_gdf_main_enriched.columns and map_value_col != 'num_clinics': hover_data_map.append('num_clinics')
            if 'facility_coverage_score' in district_gdf_main_enriched.columns and map_value_col != 'facility_coverage_score': hover_data_map.append('facility_coverage_score')
            
            # Ensure all hover_data_map columns exist in the GDF
            final_hover_data_map = [col for col in list(dict.fromkeys(hover_data_map)) if col in district_gdf_main_enriched.columns]

            district_map_figure = plot_layered_choropleth_map(
                gdf=district_gdf_main_enriched, value_col=map_value_col,
                title=f"District Map: {selected_map_metric_name}",
                id_col='zone_id', featureidkey_prefix='properties', # Assumes zone_id is in feature.properties
                color_continuous_scale=map_color_scale_theme,
                hover_cols=final_hover_data_map, # Use the filtered list
                height=app_config.MAP_PLOT_HEIGHT, # Use configured height for maps
                mapbox_style=app_config.MAPBOX_STYLE # Use configured map style
            )
            st.plotly_chart(district_map_figure, use_container_width=True)
        else: st.info("Please select a metric from the dropdown to display on the map.") # Should not happen if options exist
    else: st.warning("No metrics with valid data are available for mapping in the enriched GeoDataFrame.")
else:
    st.error("🚨 District map cannot be displayed: Enriched zone geographic data is unavailable, missing geometry, or all geometries are empty. Please check the data loading and enrichment process.")
st.markdown("---") # Visual separator

# --- Tabs for Detailed Analysis: Trends, Zonal Comparison, Intervention Insights ---
tab_titles_district = [
    "📈 District-Wide Trends", "📊 Zonal Comparative Analysis", "🎯 Intervention Planning Insights"
] # Added emojis
tab_dist_trends, tab_dist_comparison, tab_dist_interventions = st.tabs(tab_titles_district)

with tab_dist_trends:
    st.header("📈 District-Wide Health & Environmental Trends")
    # Ensure data for trends is available for the selected period
    if (filtered_health_for_trends.empty and filtered_iot_for_trends.empty):
        st.info(f"No health or environmental data available for the selected trend period ({selected_start_date_dist_trends.strftime('%d %b %Y')} to {selected_end_date_dist_trends.strftime('%d %b %Y')}). Please adjust the date range or check data sources.")
    else:
        st.markdown(f"Displaying trends from **{selected_start_date_dist_trends.strftime('%d %b %Y')}** to **{selected_end_date_dist_trends.strftime('%d %b %Y')}**.")
        
        # --- Key Disease Trends ---
        st.subheader("Key Disease Incidence Trends (New Cases Identified per Week)")
        cols_disease_trends_dist = st.columns(2)
        with cols_disease_trends_dist[0]:
            if not filtered_health_for_trends.empty and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
                tb_trend_src_df = filtered_health_for_trends[filtered_health_for_trends['condition'] == 'TB'].copy()
                weekly_new_tb_trend = get_trend_data(tb_trend_src_df, 'patient_id', date_col='date', period='W', agg_func='nunique')
                if not weekly_new_tb_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(weekly_new_tb_trend, "Weekly New TB Patients Identified", y_axis_title="New TB Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No TB trend data available for this period.")
            else: st.caption("TB trend data unavailable (missing 'condition' or 'patient_id' in health records, or no data in period).")
        with cols_disease_trends_dist[1]:
            if not filtered_health_for_trends.empty and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
                malaria_trend_src_df = filtered_health_for_trends[filtered_health_for_trends['condition'] == 'Malaria'].copy()
                weekly_new_malaria_trend = get_trend_data(malaria_trend_src_df, 'patient_id', date_col='date', period='W', agg_func='nunique')
                if not weekly_new_malaria_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(weekly_new_malaria_trend, "Weekly New Malaria Patients Identified", y_axis_title="New Malaria Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No Malaria trend data available for this period.")
            else: st.caption("Malaria trend data unavailable.")

        # --- Population Wellness & Environmental Trends ---
        st.subheader("Population Wellness & Environmental Quality Trends")
        cols_wellness_env_trends_dist = st.columns(2)
        with cols_wellness_env_trends_dist[0]: # Average Patient Steps Trend
            if not filtered_health_for_trends.empty and 'avg_daily_steps' in filtered_health_for_trends.columns:
                weekly_avg_steps_district = get_trend_data(filtered_health_for_trends, 'avg_daily_steps', date_col='date', period='W', agg_func='mean')
                if not weekly_avg_steps_district.empty:
                    st.plotly_chart(plot_annotated_line_chart(weekly_avg_steps_district, "Weekly Avg. Patient Daily Steps", y_axis_title="Average Steps", target_line=app_config.TARGET_DAILY_STEPS, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No patient steps trend data available for this period.")
            else: st.caption("Patient steps data ('avg_daily_steps') missing or unavailable in this period.")
        with cols_wellness_env_trends_dist[1]: # Average Clinic CO2 Trend
            if not filtered_iot_for_trends.empty and 'avg_co2_ppm' in filtered_iot_for_trends.columns:
                daily_avg_co2_district = get_trend_data(filtered_iot_for_trends, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not daily_avg_co2_district.empty:
                    st.plotly_chart(plot_annotated_line_chart(daily_avg_co2_district, "Daily Avg. CO2 (All Monitored Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No CO2 trend data available from clinics for this period.")
            else: st.caption("IoT data for clinic CO2 ('avg_co2_ppm') missing or unavailable in this period.")


with tab_dist_comparison:
    st.header("📊 Zonal Comparative Analysis")
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched:
        st.markdown("Visually compare zones based on key aggregated health, resource, environmental, and socio-economic metrics. Data reflects overall aggregates unless filtered.")
        
        # Define metrics for comparison table and bar chart from the enriched GDF
        comp_metrics_options = {
            "Avg. AI Risk Score": {"col": "avg_risk_score", "higher_is_worse": True, "format": "{:.1f}"},
            "Total Key Infections (Count)": {"col": "total_active_key_infections", "higher_is_worse": True, "format": "{:.0f}"},
            "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "higher_is_worse": True, "format": "{:.1f}"},
            "Facility Coverage Score (%)": {"col": "facility_coverage_score", "higher_is_worse": False, "format": "{:.1f}%"}, # Higher % is better
            "Population": {"col": "population", "format": "{:,.0f}"}, # Neutral
            "Number of Clinics": {"col": "num_clinics", "higher_is_worse": False, "format": "{:.0f}"},
            "Socio-Economic Index (Higher=Better)": {"col": "socio_economic_index", "higher_is_worse": False, "format": "{:.2f}"},
            "Avg. Patient Daily Steps": {"col": "avg_daily_steps_zone", "higher_is_worse": False, "format":"{:,.0f}"},
            "Avg. Clinic CO2 (ppm)": {"col": "zone_avg_co2", "higher_is_worse": True, "format": "{:.0f}"}
        }
        available_comp_metrics_dict = {
            disp_name: details for disp_name, details in comp_metrics_options.items()
            if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
        }

        if available_comp_metrics_dict:
            st.subheader("Zonal Statistics Overview Table")
            cols_for_comp_table = ['name'] + [d['col'] for d in available_comp_metrics_dict.values()]
            comp_table_df = district_gdf_main_enriched[[col for col in cols_for_comp_table if col in district_gdf_main_enriched.columns]].copy()
            comp_table_df.rename(columns={'name':'Zone Name'}, inplace=True)
            if 'Zone Name' in comp_table_df.columns: comp_table_df.set_index('Zone Name', inplace=True)
            
            # Prepare for styled DataFrame display
            style_format_config_comp = {details["col"]: details.get("format", "{:.1f}") # Default format
                                       for disp_name, details in available_comp_metrics_dict.items()
                                       if "format" in details and details["col"] in comp_table_df.columns}
            
            styler_comp = comp_table_df.style.format(style_format_config_comp)
            for disp_name_style, details_style in available_comp_metrics_dict.items():
                col_style_name = details_style["col"]
                if col_style_name in comp_table_df.columns:
                    cmap_to_use = 'Reds' if details_style.get("higher_is_worse", False) else 'Greens' # Red for bad-high, Green for good-high
                    try: styler_comp = styler_comp.background_gradient(subset=[col_style_name], cmap=cmap_to_use, axis=0)
                    except Exception as e_style: logger.debug(f"Styling failed for col {col_style_name}: {e_style}") # Catch minor styling issues

            st.dataframe(styler_comp, use_container_width=True, height=min(len(comp_table_df) * 45 + 50, 600)) # Dynamic height

            st.subheader("Visual Comparison by Selected Metric")
            selected_bar_metric_name_comp = st.selectbox(
                "Select metric for bar chart comparison:", list(available_comp_metrics_dict.keys()),
                key="district_comparison_barchart_selector"
            )
            selected_bar_details_comp = available_comp_metrics_dict.get(selected_bar_metric_name_comp)
            if selected_bar_details_comp:
                bar_col_comp = selected_bar_details_comp["col"]
                bar_text_format_spec = selected_bar_details_comp.get("format", "{:.1f}").replace('{','').replace('}','').split(':')[-1]
                
                st.plotly_chart(plot_bar_chart(
                    district_gdf_main_enriched, x_col='name', y_col=bar_col_comp,
                    title=f"{selected_bar_metric_name_comp} by Zone",
                    x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 150, # Taller for more zones
                    sort_values_by=bar_col_comp, ascending=selected_bar_details_comp.get("higher_is_worse", True), # Sort to show "worst" or "best" first
                    text_auto=True, text_format=bar_text_format_spec
                ), use_container_width=True)
        else: st.info("No metrics with valid data are available for Zonal Comparison in the enriched GDF.")
    else: st.info("No zonal data available for comparison. This could be due to issues in loading or enriching geographic zone data.")


with tab_dist_interventions:
    st.header("🎯 Intervention Planning Insights")
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched:
        st.markdown("Identify zones based on configurable criteria related to risk, disease burden, resource access, and environmental factors. Data reflects overall aggregates.")

        # Define criteria lambdas more robustly using .get() for safety if columns are missing
        intervention_criteria_config = {
            f"High Avg. Risk Score (≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']})":
                lambda df: df.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.RISK_THRESHOLDS['district_zone_high_risk'],
            f"Low Facility Coverage (< {app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD}%)":
                lambda df: df.get('facility_coverage_score', pd.Series(dtype=float)) < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD,
            f"High Prevalence (Top {100-app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE*100:.0f}%)": # e.g. Top 25%
                lambda df: df.get('prevalence_per_1000', pd.Series(dtype=float)) >= df.get('prevalence_per_1000', pd.Series(dtype=float)).quantile(app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE) if df.get('prevalence_per_1000', pd.Series(dtype=float)).notna().any() else pd.Series([False]*len(df), index=df.index),
            f"High TB Burden (> {app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD} cases)": # Example: Absolute cases
                lambda df: df.get('active_tb_cases', pd.Series(dtype=float)) > app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD,
            f"High Malaria Burden (> {app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD} cases)":
                lambda df: df.get('active_malaria_cases', pd.Series(dtype=float)) > app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD,
            f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)":
                lambda df: df.get('zone_avg_co2', pd.Series(dtype=float)) > app_config.CO2_LEVEL_IDEAL_PPM
        }
        # Check which criteria can be applied based on available columns in GDF
        applicable_criteria_intervention = {
            name: func for name, func in intervention_criteria_config.items()
            # Heuristic: check if any part of the name (split by space, converted to col_name format) exists in GDF cols
            if any(keyword.lower().replace("avg. ","").replace(" ","_").replace("(","").replace(")","").replace("%","").replace(">","").replace("<","").replace("=","").split("_")[0] in district_gdf_main_enriched.columns for keyword in name.split(" "))
        }

        if not applicable_criteria_intervention:
            st.warning("No criteria can be applied for intervention planning due to missing data columns in the enriched zone data. Please check data enrichment.")
        else:
            selected_intervention_criteria_names = st.multiselect(
                "Filter zones for intervention (meets ANY selected criteria):",
                options=list(applicable_criteria_intervention.keys()),
                default=list(applicable_criteria_intervention.keys())[0:min(2, len(applicable_criteria_intervention))] if applicable_criteria_intervention else [],
                key="district_intervention_criteria_multiselect",
                help="Select multiple criteria. Zones meeting any of the selected criteria will be shown."
            )

            if not selected_intervention_criteria_names:
                st.info("Select one or more criteria above to identify priority zones for potential interventions.")
            else:
                combined_filter_mask = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
                for criterion_display_name in selected_intervention_criteria_names:
                    criterion_lambda_func = applicable_criteria_intervention[criterion_display_name]
                    try:
                        mask_for_criterion = criterion_lambda_func(district_gdf_main_enriched)
                        if isinstance(mask_for_criterion, pd.Series) and mask_for_criterion.dtype == 'bool':
                             combined_filter_mask = combined_filter_mask | mask_for_criterion.fillna(False) # OR logic for multiple criteria
                        else: # If lambda did not return a boolean Series of correct length
                             logger.warning(f"Criterion '{criterion_display_name}' did not produce a valid boolean mask. Skipping.")
                    except Exception as e_inter_crit:
                        logger.error(f"Error applying intervention criterion '{criterion_display_name}': {e_inter_crit}", exc_info=True)
                        st.warning(f"Could not apply criterion: {criterion_display_name}. Error: {e_inter_crit}")
                
                priority_zones_for_intervention_df = district_gdf_main_enriched[combined_filter_mask].copy()

                if not priority_zones_for_intervention_df.empty:
                    st.markdown(f"###### Identified **{len(priority_zones_for_intervention_df)}** Zone(s) Meeting Selected Criteria:")
                    
                    # Define columns for the intervention table - more focused on actionable metrics
                    intervention_table_cols = ['name', 'population', 'avg_risk_score', 'total_active_key_infections',
                                               'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2']
                    # Include specific disease burdens if they were part of selection (optional refinement)
                    # For now, using a standard set.
                    
                    actual_intervention_table_cols = [col for col in intervention_table_cols if col in priority_zones_for_intervention_df.columns]
                    
                    # Sort identified zones: typically by highest risk, highest prevalence, or lowest coverage
                    sort_by_intervention_cols_priority = []
                    sort_ascending_flags_priority = []
                    if 'avg_risk_score' in actual_intervention_table_cols: sort_by_intervention_cols_priority.append('avg_risk_score'); sort_ascending_flags_priority.append(False) # Highest risk first
                    if 'prevalence_per_1000' in actual_intervention_table_cols: sort_by_intervention_cols_priority.append('prevalence_per_1000'); sort_ascending_flags_priority.append(False)
                    if 'facility_coverage_score' in actual_intervention_table_cols: sort_by_intervention_cols_priority.append('facility_coverage_score'); sort_ascending_flags_priority.append(True) # Lowest coverage first

                    intervention_df_sorted = priority_zones_for_intervention_df.sort_values(
                        by=sort_by_intervention_cols_priority, ascending=sort_ascending_flags_priority
                    ) if sort_by_intervention_cols_priority else priority_zones_for_intervention_df # Fallback if sort cols are missing
                    
                    st.dataframe(
                        intervention_df_sorted[actual_intervention_table_cols],
                        use_container_width=True,
                        column_config={ # Define formatting and help text for clarity
                            "name": st.column_config.TextColumn("Zone Name", help="Name of the administrative zone."),
                            "population": st.column_config.NumberColumn("Population", format="%,.0f", help="Total population in the zone."),
                            "avg_risk_score": st.column_config.ProgressColumn("Avg. Risk", format="%.1f", min_value=0, max_value=100, help="Average AI-calculated risk score for the zone's population."),
                            "total_active_key_infections": st.column_config.NumberColumn("Total Infections", format="%.0f", help="Total count of active key infectious diseases."),
                            "prevalence_per_1000": st.column_config.NumberColumn("Prevalence (/1k Pop)", format="%.1f", help="Combined prevalence of key infections per 1,000 population."),
                            "facility_coverage_score": st.column_config.NumberColumn("Facility Cov. Score (%)", format="%.1f%%", help="Composite score reflecting health facility accessibility and capacity."),
                            "zone_avg_co2": st.column_config.NumberColumn("Avg. Clinic CO2 (ppm)", format="%.0f ppm", help="Average CO2 level in clinic environments within the zone.")
                        },
                        hide_index=True
                    )
                else:
                    st.success("✅ No zones currently meet the selected high-priority criteria for intervention based on the available data.")
    else:
        st.info("No zonal data available for intervention insights. This feature requires successfully loaded and enriched geographic zone data.")
