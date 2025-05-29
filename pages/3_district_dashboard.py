# pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd # Import geopandas earlier
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    enrich_zone_geodata_with_health_aggregates,
    get_district_summary_kpis,
    get_trend_data,
    hash_geodataframe # Import the custom hash function for GDF
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_layered_choropleth_map,
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_heatmap # Keep if syndromic surveillance is implemented
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Get logger for this page

def load_css():
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: # pragma: no cover
        logger.warning(f"CSS file not found at {app_config.STYLE_CSS}. Default styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None,
    gpd.GeoDataFrame: hash_geodataframe # Use the imported function object
})
def get_district_page_data():
    logger.info("Getting district page data...")
    health_df = load_health_records()
    zone_base_gdf = load_zone_data() # This is the GDF with attributes and geometry

    if health_df.empty:
        logger.error("Health records are empty. District dashboard might be incomplete or show no data.")
        # If zone_base_gdf is also problematic, return Nones
        if zone_base_gdf is None or zone_base_gdf.empty:
            st.error("🚨 Critical Error: Could not load base health records or zone geographic data.")
            logger.error("Both health records and zone geographic data are empty/None.")
            return pd.DataFrame(), None # Return empty DF for health_df, None for GDF
        # If zone_base_gdf is fine, enrich it with an empty health_df (will add 0-value columns)
        logger.warning("Enriching zone GDF with empty health records.")
        enriched_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, health_df)
        return health_df, enriched_gdf

    if zone_base_gdf is None or zone_base_gdf.empty:
        st.error("🚨 Critical Error: Could not load zone geographic data. Map and zonal analysis will be unavailable.")
        logger.error("Zone geographic data is empty/None.")
        return health_df, None # Return health_df and None for GDF

    # Enrich the zone_base_gdf with aggregated health data
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, health_df)

    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
         st.error("🚨 Critical Error: Failed to merge health aggregates with zone geographic data. Map and zonal stats may be incorrect.")
         logger.error("Enrichment of zone GDF resulted in empty or None GDF.")
         return health_df, zone_base_gdf # Fallback to base GDF to at least show geometries if possible

    logger.info("Successfully retrieved district page data.")
    return health_df, enriched_zone_gdf

health_records_for_trends, district_map_and_zonal_stats_gdf = get_district_page_data()

# --- Main Page ---
# Ensure dataframes are not None before proceeding
if health_records_for_trends is None: health_records_for_trends = pd.DataFrame()
if district_map_and_zonal_stats_gdf is None: district_map_and_zonal_stats_gdf = gpd.GeoDataFrame()


st.title("🗺️ District Health Officer Dashboard")
st.markdown("**Strategic Overview for Population Health Management & Resource Allocation**")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("District Filters")
start_date_filter, end_date_filter = None, None # Initialize

if not health_records_for_trends.empty and 'date' in health_records_for_trends.columns:
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']): # pragma: no cover
        try:
            health_records_for_trends['date'] = pd.to_datetime(health_records_for_trends['date'], errors='coerce')
            health_records_for_trends.dropna(subset=['date'], inplace=True)
        except Exception as e_dt_conv: # pragma: no cover
            logger.error(f"Error converting 'date' column to datetime in health_records_for_trends: {e_dt_conv}")
            # If conversion fails, we might not be able to set date filters
            health_records_for_trends = pd.DataFrame() # Invalidate for date operations

    if not health_records_for_trends.empty: # Check again after potential dropna
        min_date = health_records_for_trends['date'].min().date()
        max_date = health_records_for_trends['date'].max().date()

        default_start_dt = pd.to_datetime(max_date) - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
        default_start_date = default_start_dt.date()

        if default_start_date < min_date: default_start_date = min_date
        if default_start_date > max_date: default_start_date = max_date if max_date >=min_date else min_date


        start_date_filter, end_date_filter = st.sidebar.date_input(
            "Select Date Range for Trends:",
            value=[default_start_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="district_date_range_filter_v2", # Ensure unique key
            help="This date range applies to time-series trend charts."
        )
    else: # pragma: no cover
        st.sidebar.warning("No valid date data in health records for date filter.")
else: # pragma: no cover
    st.sidebar.warning("Health records are empty or 'date' column is missing. Cannot set date filter for trends.")

# Filter health_records_for_trends for time-series charts
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not health_records_for_trends.empty:
    filtered_health_records_for_trends = health_records_for_trends[
        (health_records_for_trends['date'].dt.date >= start_date_filter) &
        (health_records_for_trends['date'].dt.date <= end_date_filter)
    ].copy()
elif not health_records_for_trends.empty:
    logger.info("Date filter for trends is invalid or not set, using all available trend data.")
    filtered_health_records_for_trends = health_records_for_trends.copy()
else:
    filtered_health_records_for_trends = pd.DataFrame()


# --- KPIs ---
if not district_map_and_zonal_stats_gdf.empty:
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf)
    st.subheader("District-Wide Key Performance Indicators (Overall)")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        pop_risk_val = district_kpis.get('avg_population_risk', 0)
        pop_risk_status = "High" if pop_risk_val > app_config.RISK_THRESHOLDS['district_zone_high_risk'] -5 else \
                          "Moderate" if pop_risk_val > app_config.RISK_THRESHOLDS['moderate'] else "Low"
        render_kpi_card("Avg. Population Risk", f"{pop_risk_val:.1f}", "🎯", status=pop_risk_status,
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols[1]:
        fac_cov_val = district_kpis.get('overall_facility_coverage', 0)
        fac_cov_status = "Low" if fac_cov_val < 60 else "Moderate" if fac_cov_val < 80 else "High"
        render_kpi_card("Facility Coverage Score", f"{fac_cov_val:.1f}%", "🏥", status=fac_cov_status,
                        help_text="Population-weighted average facility coverage score (considers access & capacity).")
    with kpi_cols[2]:
        hz_val = district_kpis.get('zones_high_risk', 0)
        hz_status = "High" if hz_val > (len(district_map_and_zonal_stats_gdf) * 0.25) else \
                    "Moderate" if hz_val > 0 else "Low"
        render_kpi_card("High-Risk Zones", str(hz_val), "⚠️", status=hz_status,
                        help_text=f"Number of zones with average risk score >= {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
else: # pragma: no cover
    st.warning("Zone map data not available for KPIs. Ensure zone data is loaded correctly.")
st.markdown("---")

# --- Choropleth Map ---
if not district_map_and_zonal_stats_gdf.empty:
    st.subheader("Interactive Health Map: Risk & Resources by Zone")
    map_metric_options = {
        "Average AI Risk Score": "avg_risk_score", "Active Cases (Count)": "active_cases",
        "Prevalence per 1,000": "prevalence_per_1000", "Facility Coverage Score": "facility_coverage_score",
        "Population": "population", "Socio-Economic Index": "socio_economic_index",
        "Number of Clinics": "num_clinics", "Avg. Travel Time to Clinic (min)": "avg_travel_time_clinic_min"
    }
    available_map_metrics = {k: v for k, v in map_metric_options.items() if v in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v].notna().any()}
    
    if available_map_metrics:
        selected_map_metric_display = st.selectbox(
            "Select Metric to Display on Map:", list(available_map_metrics.keys()), 
            key="district_map_metric_select_v2",
            help="Choose a metric to visualize spatially across the district zones."
        )
        selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)

        if selected_map_metric_col:
            color_scale = "OrRd" 
            if any(keyword in selected_map_metric_col.lower() for keyword in ["coverage", "socio_economic", "clinics"]):
                color_scale = "Mint" 
            elif "travel_time" in selected_map_metric_col.lower(): color_scale = "OrRd" 

            map_figure = plot_layered_choropleth_map(
                district_map_and_zonal_stats_gdf, value_col=selected_map_metric_col,
                title=f"{selected_map_metric_display} by Zone", id_col='zone_id', 
                featureidkey_prop='zone_id', color_continuous_scale=color_scale,
                hover_cols=['name', 'population', selected_map_metric_col, 'num_clinics', 'avg_travel_time_clinic_min'],
                height=app_config.MAP_PLOT_HEIGHT, zoom_level=app_config.MAP_DEFAULT_ZOOM
            )
            st.plotly_chart(map_figure, use_container_width=True)
        else: # pragma: no cover
            st.warning("No metric selected for the map or selected metric column data is all NaNs.")
    else:
        st.warning("No metrics with valid data available for mapping in the current zone data.")
else: # pragma: no cover
    st.warning("Zone map data is empty. Cannot display map.")
st.markdown("---")

# --- Tabs for Trends and Comparative Analysis ---
tab_trends, tab_comparison, tab_interventions = st.tabs(["📈 District Trends", "📊 Zonal Comparison", "🛠️ Intervention Insights"])

with tab_trends:
    if not filtered_health_records_for_trends.empty and start_date_filter and end_date_filter:
        st.subheader(f"Key Health Trends ({start_date_filter.strftime('%d %b %Y')} - {end_date_filter.strftime('%d %b %Y')})")
        trend_cols = st.columns(2)
        with trend_cols[0]:
            overall_risk_trend = get_trend_data(filtered_health_records_for_trends, 'ai_risk_score', period='W', agg_func='mean')
            if not overall_risk_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    overall_risk_trend, "Weekly Avg. AI Risk Score (District)", y_axis_title="Avg. Risk Score",
                    target_line=app_config.TARGET_PATIENT_RISK_SCORE, target_label=f"Target Risk: {app_config.TARGET_PATIENT_RISK_SCORE}",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else: st.caption("No data for risk score trend in selected period.")
        
        with trend_cols[1]:
            # Use a copy for adding flags to avoid modifying cached df
            trend_df_copy = filtered_health_records_for_trends.copy()
            trend_df_copy['new_case_flag'] = trend_df_copy['condition'].astype(str).isin(['TB', 'Malaria', 'ARI']).astype(int)
            new_cases_trend = get_trend_data(trend_df_copy, 'new_case_flag', period='W', agg_func='sum')

            if not new_cases_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    new_cases_trend, "Weekly New Cases (Key Conditions)", y_axis_title="Number of New Cases",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else: st.caption("No data for new cases trend in selected period.")
    else:
        st.info("No data available for trend analysis based on current filters or data load for selected period.")

with tab_comparison:
    if not district_map_and_zonal_stats_gdf.empty:
        st.subheader("Comparative Analysis Across Zones (Overall Aggregates)")
        comparison_metrics_options = {
            "Avg. AI Risk Score": "avg_risk_score", "Prevalence per 1,000": "prevalence_per_1000",
            "Facility Coverage Score": "facility_coverage_score", "Population": "population",
            "Number of Clinics": "num_clinics", "Socio-Economic Index": "socio_economic_index",
            "CHW Visits in Zone": "chw_visits_in_zone", "Total Tests Conducted": "total_tests_conducted"
        }
        available_comp_metrics = {k: v for k,v in comparison_metrics_options.items() if v in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v].notna().any()}

        if available_comp_metrics:
            display_comp_cols = ['name'] + [col for col in available_comp_metrics.values() if col != 'name'] # Ensure 'name' is first and unique
            display_comp_cols = list(dict.fromkeys(display_comp_cols)) # Keep order, remove duplicates

            comparison_df_display = district_map_and_zonal_stats_gdf[display_comp_cols].copy()
            comparison_df_display.set_index('name', inplace=True)
            
            format_dict = {col: "{:.2f}" for col in available_comp_metrics.values() if district_map_and_zonal_stats_gdf[col].dtype == 'float64'}
            format_dict.update({
                'population': "{:,.0f}", 'num_clinics': "{:.0f}", 
                'chw_visits_in_zone': "{:,.0f}", 'total_tests_conducted': "{:,.0f}"
            })
            # Filter format_dict for existing columns in comparison_df_display
            valid_format_dict = {k:v for k,v in format_dict.items() if k in comparison_df_display.columns}


            st.dataframe(
                comparison_df_display.style.format(valid_format_dict).background_gradient(
                    cmap='RdYlGn_r', subset=[m for m in available_comp_metrics.values() if m not in ['population', 'name', 'num_clinics', 'chw_visits_in_zone', 'total_tests_conducted', 'socio_economic_index']]
                ).highlight_max(subset=['facility_coverage_score', 'socio_economic_index', 'num_clinics'], color='lightgreen'
                ).highlight_min(subset=['avg_risk_score', 'prevalence_per_1000'], color='#FFCCCB'),
                use_container_width=True, height=300 
            )
            
            selected_comp_metric_display = st.selectbox("Select metric for bar chart comparison:", list(available_comp_metrics.keys()), key="district_comp_metric_bar_v3")
            selected_comp_metric_col = available_comp_metrics.get(selected_comp_metric_display)

            if selected_comp_metric_col:
                sort_ascending_bar = any(keyword in selected_comp_metric_col.lower() for keyword in ["coverage", "socio_economic", "clinics"])
                
                bar_df_comp = district_map_and_zonal_stats_gdf[['name', selected_comp_metric_col]].copy()
                # Sorting is now handled by plot_bar_chart's sort_values_by parameter
                st.plotly_chart(plot_bar_chart(
                    bar_df_comp, x_col='name', y_col=selected_comp_metric_col, 
                    title=f"{selected_comp_metric_display} by Zone", x_axis_title="Zone Name",
                    height=app_config.DEFAULT_PLOT_HEIGHT + 70, # Taller for potentially many zones
                    sort_values_by=selected_comp_metric_col, ascending=sort_ascending_bar,
                    text_auto = '.2s' if 'population' in selected_comp_metric_col else True # Abbreviate large numbers
                ), use_container_width=True)
        else: # pragma: no cover
            st.info("No metrics with valid data available for Zonal Comparison.")
    else: # pragma: no cover
        st.info("No zonal data available for comparison.")


with tab_interventions:
    if not district_map_and_zonal_stats_gdf.empty:
        st.subheader("Intervention Planning Insights")
        st.markdown("Identify zones based on combined risk factors. Data shown aggregates all available historical records.")
        
        priority_criteria_list = []
        if 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns:
            priority_criteria_list.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
        if 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns:
            priority_criteria_list.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < 50) 
        if 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns and not district_map_and_zonal_stats_gdf['prevalence_per_1000'].empty:
            prevalence_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
            if pd.notna(prevalence_q75) and prevalence_q75 > 0: # Ensure q75 is valid and not zero
                 priority_criteria_list.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prevalence_q75)

        if priority_criteria_list:
            # Combine criteria: a zone is priority if it meets ANY of the criteria
            try:
                final_priority_mask = pd.concat(priority_criteria_list, axis=1).any(axis=1)
                priority_zones_df = district_map_and_zonal_stats_gdf[final_priority_mask]
            except Exception as e_concat: # pragma: no cover
                logger.error(f"Error during priority criteria concatenation: {e_concat}")
                priority_zones_df = pd.DataFrame() # Fallback to empty
            
            if not priority_zones_df.empty:
                st.markdown("###### Zones Identified for Potential Intervention (All-Time Aggregates):")
                intervention_cols = ['name', 'population', 'avg_risk_score', 'prevalence_per_1000', 'facility_coverage_score', 'num_clinics']
                actual_intervention_cols = [col for col in intervention_cols if col in priority_zones_df.columns]
                st.dataframe(
                    priority_zones_df[actual_intervention_cols].sort_values(by='avg_risk_score', ascending=False),
                    use_container_width=True,
                    column_config={ 
                        "population": st.column_config.NumberColumn(format="%d"),
                        "avg_risk_score": st.column_config.NumberColumn(format="%.1f"),
                        "prevalence_per_1000": st.column_config.NumberColumn(format="%.1f"),
                        "facility_coverage_score": st.column_config.NumberColumn(format="%.1f%%"),
                        "num_clinics": st.column_config.NumberColumn(format="%d"),
                    }
                )
            else:
                st.success("✅ No zones currently meet the defined high-priority criteria for intervention based on overall data.")
        else: # pragma: no cover
            st.info("No priority criteria applicable or data available for intervention planning suggestions.")
    else: # pragma: no cover
        st.info("No zonal data available for intervention insights.")
