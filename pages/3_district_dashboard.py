# pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records, # Still needed for overall min/max dates
    load_zone_data,
    enrich_zone_geodata_with_health_aggregates,
    get_district_summary_kpis,
    get_trend_data
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_layered_choropleth_map,
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_heatmap
)
import logging # Import logging

# --- Page Configuration and Styling ---
st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) # Get logger for this page

def load_css():
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty else None, # Handle empty df for hashing
    gpd.GeoDataFrame: "hash_geodataframe" # Assuming hash_geodataframe is globally accessible or defined here
})
def get_district_page_data():
    # Define hash_geodataframe here if not imported or globally available from core_data_processing
    # For simplicity, I'll assume it's accessible or you'd import it if it was in a shared utils
    # from utils.core_data_processing import hash_geodataframe # If it's there

    health_df = load_health_records()
    zone_base_gdf = load_zone_data()
    
    if health_df.empty:
        logger.error("Health records are empty. District dashboard might be incomplete.")
        # Return empty GDF if base GDF is also problematic or proceed with what we have
        if zone_base_gdf is None or zone_base_gdf.empty:
            st.error("🚨 Critical Error: Could not load base health records or zone geographic data.")
            return pd.DataFrame(), None # Return empty DF and None
        # If zone_base_gdf is fine, enrich it with an empty health_df (will add 0-value columns)
        enriched_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, health_df)
        return health_df, enriched_gdf

    if zone_base_gdf is None or zone_base_gdf.empty:
        st.error("🚨 Critical Error: Could not load zone geographic data.")
        return health_df, None # Return health_df and None for GDF

    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, health_df)
    
    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
         st.error("🚨 Critical Error: Failed to merge health aggregates with zone geographic data.")
         return health_df, zone_base_gdf # Fallback to base GDF

    return health_df, enriched_zone_gdf

health_records_for_trends, district_map_and_zonal_stats_gdf = get_district_page_data()

# --- Main Page ---
if health_records_for_trends is None or district_map_and_zonal_stats_gdf is None :
    st.error("Essential data could not be loaded. District Dashboard cannot be fully displayed.")
    if health_records_for_trends is None: health_records_for_trends = pd.DataFrame() # Ensure it's a DF for safety
    if district_map_and_zonal_stats_gdf is None: district_map_and_zonal_stats_gdf = gpd.GeoDataFrame() # Ensure it's a GDF
    # Allow app to proceed with partial data if possible, or st.stop()
    if health_records_for_trends.empty and district_map_and_zonal_stats_gdf.empty:
        st.stop()

st.title("🗺️ District Health Officer Dashboard")
st.markdown("**Strategic Overview for Population Health Management & Resource Allocation**")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("District Filters")

# Handle case where health_records_for_trends might be empty after loading attempts
if not health_records_for_trends.empty and 'date' in health_records_for_trends.columns:
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']):
        health_records_for_trends['date'] = pd.to_datetime(health_records_for_trends['date'], errors='coerce')
        health_records_for_trends.dropna(subset=['date'], inplace=True)

    if not health_records_for_trends.empty:
        min_date = health_records_for_trends['date'].min().date()
        max_date = health_records_for_trends['date'].max().date()

        default_start_dt = pd.to_datetime(max_date) - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
        default_start_date = default_start_dt.date()

        # Ensure default_start_date is not before min_date
        if default_start_date < min_date:
            default_start_date = min_date
        
        # Ensure default_start_date is not after max_date (can happen if date range in data is small)
        if default_start_date > max_date:
            default_start_date = max_date # Or min_date, depending on desired behavior

        start_date_filter, end_date_filter = st.sidebar.date_input(
            "Select Date Range for Trends:",
            value=[default_start_date, max_date], # Both are datetime.date objects
            min_value=min_date,
            max_value=max_date,
            key="district_date_range_filter",
            help="This date range applies to time-series trend charts."
        )
    else:
        st.sidebar.warning("No valid date data in health records for date filter.")
        start_date_filter, end_date_filter = None, None # Fallback
else:
    st.sidebar.warning("Health records are empty or 'date' column is missing. Cannot set date filter.")
    start_date_filter, end_date_filter = None, None # Fallback

# Filter health_records_for_trends for time-series charts
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter:
    filtered_health_records_for_trends = health_records_for_trends[
        (health_records_for_trends['date'].dt.date >= start_date_filter) &
        (health_records_for_trends['date'].dt.date <= end_date_filter)
    ].copy()
elif not health_records_for_trends.empty: # If filters are invalid/None but data exists
    logger.warning("Date filter for trends is invalid or not set, using all available trend data.")
    filtered_health_records_for_trends = health_records_for_trends.copy()
else: # No data to filter
    filtered_health_records_for_trends = pd.DataFrame()


# --- KPIs (from district_map_and_zonal_stats_gdf which has all-time aggregates) ---
if not district_map_and_zonal_stats_gdf.empty:
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf)
    st.subheader("District-Wide Key Performance Indicators (Overall)")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        pop_risk_status = "High" if district_kpis.get('avg_population_risk', 0) > app_config.RISK_THRESHOLDS['district_zone_high_risk'] -5 else \
                          "Moderate" if district_kpis.get('avg_population_risk', 0) > app_config.RISK_THRESHOLDS['moderate'] else "Low"
        render_kpi_card("Avg. Population Risk", f"{district_kpis.get('avg_population_risk', 0):.1f}", "🎯", status=pop_risk_status,
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols[1]:
        fac_cov_status = "Low" if district_kpis.get('overall_facility_coverage', 0) < 60 else \
                         "Moderate" if district_kpis.get('overall_facility_coverage', 0) < 80 else "High"
        render_kpi_card("Facility Coverage Score", f"{district_kpis.get('overall_facility_coverage', 0):.1f}%", "🏥", status=fac_cov_status,
                        help_text="Population-weighted average facility coverage score (considers access & capacity).")
    with kpi_cols[2]:
        hz_status = "High" if district_kpis.get('zones_high_risk', 0) > (len(district_map_and_zonal_stats_gdf) * 0.25) else \
                    "Moderate" if district_kpis.get('zones_high_risk', 0) > 0 else "Low"
        render_kpi_card("High-Risk Zones", str(district_kpis.get('zones_high_risk', 0)), "⚠️", status=hz_status,
                        help_text=f"Number of zones with average risk score >= {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
else:
    st.warning("Zone map data not available for KPIs.")
st.markdown("---")

# --- Choropleth Map ---
# (Rest of the District Dashboard code remains the same as the "complete corrected file" version)
# ... ensure that district_map_and_zonal_stats_gdf and filtered_health_records_for_trends
# are checked for emptiness before being used by plotting functions ...
# Example:
if not district_map_and_zonal_stats_gdf.empty:
    st.subheader("Interactive Health Map: Risk & Resources by Zone")
    map_metric_options = {
        "Average AI Risk Score": "avg_risk_score",
        "Active Cases (Count)": "active_cases",
        "Prevalence per 1,000": "prevalence_per_1000",
        "Facility Coverage Score": "facility_coverage_score",
        "Population": "population",
        "Socio-Economic Index": "socio_economic_index",
        "Number of Clinics": "num_clinics",
        "Avg. Travel Time to Clinic (min)": "avg_travel_time_clinic_min"
    }
    available_map_metrics = {k: v for k, v in map_metric_options.items() if v in district_map_and_zonal_stats_gdf.columns}
    
    if available_map_metrics:
        selected_map_metric_display = st.selectbox(
            "Select Metric to Display on Map:", 
            list(available_map_metrics.keys()), 
            key="district_map_metric_select",
            help="Choose a metric to visualize spatially across the district zones."
        )
        selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)

        if selected_map_metric_col:
            # ... (map plotting logic as before) ...
            color_scale = "OrRd" 
            if any(keyword in selected_map_metric_col.lower() for keyword in ["coverage", "socio_economic", "clinics"]):
                color_scale = "Mint" 
            elif "travel_time" in selected_map_metric_col.lower():
                color_scale = "OrRd" 

            map_figure = plot_layered_choropleth_map(
                district_map_and_zonal_stats_gdf,
                value_col=selected_map_metric_col,
                title=f"{selected_map_metric_display} by Zone",
                id_col='zone_id', 
                featureidkey_prop='zone_id',
                color_continuous_scale=color_scale,
                hover_cols=['name', 'population', selected_map_metric_col, 'num_clinics', 'avg_travel_time_clinic_min'],
                height=app_config.MAP_PLOT_HEIGHT,
                zoom_level=app_config.MAP_DEFAULT_ZOOM
            )
            st.plotly_chart(map_figure, use_container_width=True)
        else:
            st.warning("No metric selected for the map or selected metric column is missing.")
    else:
        st.warning("No metrics available for mapping in the current zone data.")
else:
    st.warning("Zone map data is empty. Cannot display map.")

st.markdown("---")

# --- Tabs for Trends and Comparative Analysis ---
tab_trends, tab_comparison, tab_interventions = st.tabs(["📈 District Trends", "📊 Zonal Comparison", "🛠️ Intervention Insights"])

with tab_trends:
    if not filtered_health_records_for_trends.empty:
        st.subheader(f"Key Health Trends ({start_date_filter.strftime('%d %b %Y') if start_date_filter else 'All time'} - {end_date_filter.strftime('%d %b %Y') if end_date_filter else ''})")
        # ... (trend plotting logic as before, using filtered_health_records_for_trends) ...
        trend_cols = st.columns(2)
        with trend_cols[0]:
            overall_risk_trend = get_trend_data(filtered_health_records_for_trends, 'ai_risk_score', period='W')
            if not overall_risk_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    overall_risk_trend, "Weekly Avg. AI Risk Score (District)", y_axis_title="Avg. Risk Score",
                    target_line=app_config.TARGET_PATIENT_RISK_SCORE, target_label=f"Target Risk: {app_config.TARGET_PATIENT_RISK_SCORE}",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else: st.caption("No data for risk score trend in selected period.")
        
        with trend_cols[1]:
            filtered_health_records_for_trends['new_case_flag'] = filtered_health_records_for_trends['condition'].astype(str).isin(['TB', 'Malaria', 'ARI']).astype(int)
            new_cases_trend = filtered_health_records_for_trends.groupby(pd.Grouper(key='date', freq='W'))['new_case_flag'].sum()
            new_cases_trend_series = pd.Series(new_cases_trend.values, index=new_cases_trend.index, name="New Cases")

            if not new_cases_trend_series.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    new_cases_trend_series, "Weekly New Cases (Key Conditions)", y_axis_title="Number of New Cases",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else: st.caption("No data for new cases trend in selected period.")
    else:
        st.info("No data available for trend analysis based on current filters or data load.")

# ... (Rest of the tabs, ensuring checks for district_map_and_zonal_stats_gdf.empty before use)
with tab_comparison:
    if not district_map_and_zonal_stats_gdf.empty:
        st.subheader("Comparative Analysis Across Zones (Overall Aggregates)")
        # ... (comparison logic as before) ...
        comparison_metrics_options = {
            "Avg. AI Risk Score": "avg_risk_score", "Prevalence per 1,000": "prevalence_per_1000",
            "Facility Coverage Score": "facility_coverage_score", "Population": "population",
            "Number of Clinics": "num_clinics", "Socio-Economic Index": "socio_economic_index"
        }
        available_comp_metrics = {k: v for k,v in comparison_metrics_options.items() if v in district_map_and_zonal_stats_gdf.columns}

        if available_comp_metrics:
            display_comp_cols = ['name'] + list(available_comp_metrics.values())
            comparison_df_display = district_map_and_zonal_stats_gdf[display_comp_cols].copy()
            comparison_df_display.set_index('name', inplace=True)
            
            format_dict = {col: "{:.2f}" for col in available_comp_metrics.values() if district_map_and_zonal_stats_gdf[col].dtype == 'float64'}
            format_dict['population'] = "{:,}"
            format_dict['num_clinics'] = "{:.0f}"

            st.dataframe(
                comparison_df_display.style.format(format_dict).background_gradient(
                    cmap='RdYlGn_r', subset=[m for m in available_comp_metrics.values() if m not in ['population', 'name', 'num_clinics']]
                ).highlight_max(subset=['facility_coverage_score', 'socio_economic_index', 'num_clinics'], color='lightgreen'
                ).highlight_min(subset=['avg_risk_score', 'prevalence_per_1000'], color='#FFCCCB'),
                use_container_width=True
            )
            
            selected_comp_metric_display = st.selectbox("Select metric for bar chart comparison:", list(available_comp_metrics.keys()), key="district_comp_metric_bar_2") # Changed key
            selected_comp_metric_col = available_comp_metrics.get(selected_comp_metric_display)

            if selected_comp_metric_col:
                # Determine sort order: higher is better for coverage/socio-economic, lower is better for risk/prevalence
                sort_ascending_bar = any(keyword in selected_comp_metric_col.lower() for keyword in ["coverage", "socio_economic", "clinics"])
                
                bar_df = district_map_and_zonal_stats_gdf[['name', selected_comp_metric_col]].sort_values(
                    by=selected_comp_metric_col, ascending=sort_ascending_bar
                )
                st.plotly_chart(plot_bar_chart(
                    bar_df, x_col='name', y_col=selected_comp_metric_col, 
                    title=f"{selected_comp_metric_display} by Zone",
                    x_axis_title="Zone Name",
                    height=app_config.DEFAULT_PLOT_HEIGHT + 50,
                    sort_values_by=selected_comp_metric_col, # Let bar chart handle sorting based on its parameter
                    ascending=sort_ascending_bar
                ), use_container_width=True)
        else:
            st.info("No metrics available for Zonal Comparison.")
    else:
        st.info("No zonal data available for comparison.")


with tab_interventions:
    if not district_map_and_zonal_stats_gdf.empty:
        st.subheader("Intervention Planning Insights")
        # ... (intervention logic as before) ...
        priority_criteria_list = [] # Renamed to avoid conflict
        if 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns:
            priority_criteria_list.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
        if 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns:
            priority_criteria_list.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < 50) 
        if 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns and not district_map_and_zonal_stats_gdf['prevalence_per_1000'].empty:
            prevalence_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
            if pd.notna(prevalence_q75): # Ensure quantile is valid
                 priority_criteria_list.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prevalence_q75)

        if priority_criteria_list:
            final_priority_mask = pd.concat(priority_criteria_list, axis=1).any(axis=1)
            priority_zones_df = district_map_and_zonal_stats_gdf[final_priority_mask]
            
            if not priority_zones_df.empty:
                st.markdown("###### Zones Identified for Potential Intervention:")
                # ... (dataframe display logic as before) ...
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
                    }
                )
            else:
                st.success("✅ No zones currently meet the defined high-priority criteria for intervention.")
        else:
            st.info("No criteria defined or data available for intervention planning.")
    else:
        st.info("No zonal data available for intervention insights.")
