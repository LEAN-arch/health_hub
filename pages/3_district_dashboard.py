# pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import os
from config import app_config
from utils.core_data_processing import (
    load_health_records,
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
    plot_heatmap # Keep if syndromic surveillance is implemented
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")

def load_css():
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# --- Data Loading ---
@st.cache_data(ttl=3600) # Cache the combined data loading logic
def get_district_page_data():
    health_df = load_health_records()
    zone_base_gdf = load_zone_data() # This is the GDF with attributes and geometry
    
    if health_df.empty or zone_base_gdf is None or zone_base_gdf.empty:
        st.error("🚨 Critical Error: Could not load base health records or zone geographic data. Dashboard cannot proceed.")
        return None, None 
        
    # Enrich the zone_base_gdf with aggregated health data
    # This enriched_gdf will be used for map and current zonal stats
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, health_df)
    
    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
         st.error("🚨 Critical Error: Failed to merge health aggregates with zone geographic data.")
         return health_df, None # Return health_df for trends if it loaded

    return health_df, enriched_zone_gdf

# health_records_for_trends contains all historical data for time series
# district_map_and_zonal_stats_gdf contains the geo-data enriched with all-time health aggregates
health_records_for_trends, district_map_and_zonal_stats_gdf = get_district_page_data()


# --- Main Page ---
if health_records_for_trends is None or district_map_and_zonal_stats_gdf is None:
    # Error message already shown in get_district_page_data or its sub-functions
    st.stop() # Stop execution if essential data is missing
else:
    st.title("🗺️ District Health Officer Dashboard")
    st.markdown("**Strategic Overview for Population Health Management & Resource Allocation**")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("District Filters")
    min_date = health_records_for_trends['date'].min().date()
    max_date = health_records_for_trends['date'].max().date()
    
    start_date_filter, end_date_filter = st.sidebar.date_input(
        "Select Date Range for Trends:",
        [max_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1), max_date],
        min_value=min_date,
        max_value=max_date,
        key="district_date_range_filter",
        help="This date range applies to time-series trend charts."
    )

    # Filter health_records_for_trends for time-series charts
    if start_date_filter and end_date_filter and start_date_filter <= end_date_filter:
        filtered_health_records_for_trends = health_records_for_trends[
            (health_records_for_trends['date'].dt.date >= start_date_filter) &
            (health_records_for_trends['date'].dt.date <= end_date_filter)
        ].copy()
    else:
        st.sidebar.error("Invalid date range for trends. Please select a valid range.")
        # Fallback to using all data for trends if range is invalid, or could show no trends
        filtered_health_records_for_trends = health_records_for_trends.copy()


    # --- KPIs (from district_map_and_zonal_stats_gdf which has all-time aggregates) ---
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf)
    st.subheader("District-Wide Key Performance Indicators (Overall)")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        pop_risk_status = "High" if district_kpis['avg_population_risk'] > app_config.RISK_THRESHOLDS['district_zone_high_risk'] -5 else \
                          "Moderate" if district_kpis['avg_population_risk'] > app_config.RISK_THRESHOLDS['moderate'] else "Low"
        render_kpi_card("Avg. Population Risk", f"{district_kpis['avg_population_risk']:.1f}", "🎯", status=pop_risk_status,
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols[1]:
        fac_cov_status = "Low" if district_kpis['overall_facility_coverage'] < 60 else \
                         "Moderate" if district_kpis['overall_facility_coverage'] < 80 else "High"
        render_kpi_card("Facility Coverage Score", f"{district_kpis['overall_facility_coverage']:.1f}%", "🏥", status=fac_cov_status,
                        help_text="Population-weighted average facility coverage score (considers access & capacity).")
    with kpi_cols[2]:
        hz_status = "High" if district_kpis['zones_high_risk'] > (len(district_map_and_zonal_stats_gdf) * 0.25) else \
                    "Moderate" if district_kpis['zones_high_risk'] > 0 else "Low" # High if >25% zones are high risk
        render_kpi_card("High-Risk Zones", str(district_kpis['zones_high_risk']), "⚠️", status=hz_status,
                        help_text=f"Number of zones with average risk score >= {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    st.markdown("---")

    # --- Choropleth Map ---
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
    # Filter options to only include columns present in the GDF
    available_map_metrics = {k: v for k, v in map_metric_options.items() if v in district_map_and_zonal_stats_gdf.columns}
    
    selected_map_metric_display = st.selectbox(
        "Select Metric to Display on Map:", 
        list(available_map_metrics.keys()), 
        key="district_map_metric_select",
        help="Choose a metric to visualize spatially across the district zones."
    )
    selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)

    if selected_map_metric_col:
        color_scale = "OrRd" # Default: Oranges to Reds (higher is generally worse for risk/cases)
        if any(keyword in selected_map_metric_col.lower() for keyword in ["coverage", "socio_economic", "clinics"]):
            color_scale = "Mint" # Higher is better (Greens)
        elif "travel_time" in selected_map_metric_col.lower():
             color_scale = "OrRd" # Higher travel time is worse

        map_figure = plot_layered_choropleth_map(
            district_map_and_zonal_stats_gdf,
            value_col=selected_map_metric_col,
            title=f"{selected_map_metric_display} by Zone",
            id_col='zone_id', # Column in GDF used for 'locations' in choropleth_mapbox
            featureidkey_prop='zone_id', # Property in GeoJSON features for linking
            color_continuous_scale=color_scale,
            hover_cols=['name', 'population', selected_map_metric_col, 'num_clinics', 'avg_travel_time_clinic_min'],
            height=app_config.MAP_PLOT_HEIGHT,
            zoom_level=app_config.MAP_DEFAULT_ZOOM
        )
        st.plotly_chart(map_figure, use_container_width=True)
    else:
        st.warning(f"Metric '{selected_map_metric_display}' not available or not found in data. Check data processing.")
        
    st.markdown("---")
    
    # --- Tabs for Trends and Comparative Analysis ---
    tab_trends, tab_comparison, tab_interventions = st.tabs(["📈 District Trends", "📊 Zonal Comparison", "🛠️ Intervention Insights"])

    with tab_trends:
        st.subheader(f"Key Health Trends ({start_date_filter.strftime('%d %b')} - {end_date_filter.strftime('%d %b %Y')})")
        trend_cols = st.columns(2)
        with trend_cols[0]:
            overall_risk_trend = get_trend_data(filtered_health_records_for_trends, 'ai_risk_score', period='W')
            if not overall_risk_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    overall_risk_trend, "Weekly Avg. AI Risk Score (District)", y_axis_title="Avg. Risk Score",
                    target_line=app_config.TARGET_PATIENT_RISK_SCORE, target_label=f"Target Risk: {app_config.TARGET_PATIENT_RISK_SCORE}",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else: st.caption("No data for risk score trend.")
        
        with trend_cols[1]:
            filtered_health_records_for_trends['new_case_flag'] = filtered_health_records_for_trends['condition'].isin(['TB', 'Malaria', 'ARI']).astype(int) # Example important conditions
            new_cases_trend = filtered_health_records_for_trends.groupby(pd.Grouper(key='date', freq='W'))['new_case_flag'].sum()
            new_cases_trend_series = pd.Series(new_cases_trend.values, index=new_cases_trend.index, name="New Cases")

            if not new_cases_trend_series.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    new_cases_trend_series, "Weekly New Cases (Key Conditions)", y_axis_title="Number of New Cases",
                    height=app_config.DEFAULT_PLOT_HEIGHT
                ), use_container_width=True)
            else: st.caption("No data for new cases trend.")

    with tab_comparison:
        st.subheader("Comparative Analysis Across Zones (Overall Aggregates)")
        if not district_map_and_zonal_stats_gdf.empty:
            comparison_metrics_options = {
                "Avg. AI Risk Score": "avg_risk_score",
                "Prevalence per 1,000": "prevalence_per_1000",
                "Facility Coverage Score": "facility_coverage_score",
                "Population": "population",
                "Number of Clinics": "num_clinics",
                "Socio-Economic Index": "socio_economic_index"
            }
            available_comp_metrics = {k: v for k,v in comparison_metrics_options.items() if v in district_map_and_zonal_stats_gdf.columns}

            # Data table for comparison
            display_comp_cols = ['name'] + list(available_comp_metrics.values())
            comparison_df_display = district_map_and_zonal_stats_gdf[display_comp_cols].copy()
            comparison_df_display.set_index('name', inplace=True)
            
            # Formatting for display
            format_dict = {col: "{:.2f}" for col in available_comp_metrics.values() if district_map_and_zonal_stats_gdf[col].dtype == 'float64'}
            format_dict['population'] = "{:,}" # Comma separator for population
            format_dict['num_clinics'] = "{:.0f}"

            st.dataframe(
                comparison_df_display.style.format(format_dict).background_gradient(
                    cmap='RdYlGn_r', 
                    subset=[m for m in available_comp_metrics.values() if m not in ['population', 'name', 'num_clinics']] # Apply gradient to risk-like metrics
                ).highlight_max(subset=['facility_coverage_score', 'socio_economic_index', 'num_clinics'], color='lightgreen'
                ).highlight_min(subset=['avg_risk_score', 'prevalence_per_1000'], color='#FFCCCB'), # Light red
                use_container_width=True
            )
            
            # Bar chart for a selected metric
            selected_comp_metric_display = st.selectbox("Select metric for bar chart comparison:", list(available_comp_metrics.keys()), key="district_comp_metric_bar")
            selected_comp_metric_col = available_comp_metrics.get(selected_comp_metric_display)

            if selected_comp_metric_col:
                bar_df = district_map_and_zonal_stats_gdf[['name', selected_comp_metric_col]].sort_values(by=selected_comp_metric_col, ascending=False if "coverage" not in selected_comp_metric_col else True)
                st.plotly_chart(plot_bar_chart(
                    bar_df, x_col='name', y_col=selected_comp_metric_col, 
                    title=f"{selected_comp_metric_display} by Zone",
                    x_axis_title="Zone Name",
                    height=app_config.DEFAULT_PLOT_HEIGHT + 50
                ), use_container_width=True)
        else:
            st.info("No zonal data available for comparison.")
            
    with tab_interventions:
        st.subheader("Intervention Planning Insights")
        st.markdown("""
        This section provides data to support intervention planning. Focus on zones with high risk, low coverage, or concerning trends.
        """)
        
        # Identify priority zones
        priority_criteria = []
        if 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns:
            priority_criteria.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
        if 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns:
            priority_criteria.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < 50) # Example: coverage < 50%
        if 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns:
             # High prevalence (e.g., top 25th percentile or absolute value)
            if not district_map_and_zonal_stats_gdf['prevalence_per_1000'].empty:
                prevalence_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
                priority_criteria.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prevalence_q75)

        if priority_criteria:
            # Combine criteria: a zone is priority if it meets ANY of the criteria
            final_priority_mask = pd.concat(priority_criteria, axis=1).any(axis=1)
            priority_zones_df = district_map_and_zonal_stats_gdf[final_priority_mask]
            
            if not priority_zones_df.empty:
                st.markdown("###### Zones Identified for Potential Intervention:")
                intervention_cols = ['name', 'population', 'avg_risk_score', 'prevalence_per_1000', 'facility_coverage_score', 'num_clinics']
                actual_intervention_cols = [col for col in intervention_cols if col in priority_zones_df.columns]
                st.dataframe(
                    priority_zones_df[actual_intervention_cols].sort_values(by='avg_risk_score', ascending=False),
                    use_container_width=True,
                    column_config={ # Example to make table more readable
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

        # Conceptual placeholder for resource allocation simulation (future feature)
        st.markdown("---")
        st.markdown("###### Conceptual Resource Allocation (Future Feature)")
        st.caption("Imagine a tool here to simulate deploying resources (e.g., mobile clinics, additional CHWs) to priority zones and estimating impact.")
