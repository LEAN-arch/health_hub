import streamlit as st
import pandas as pd
from utils.data_processor import load_health_data, load_geojson_data, merge_health_data_with_geojson, get_district_summary_kpis, get_trend_data
from utils.viz_helper import render_kpi_card, plot_layered_choropleth_map, plot_annotated_line_chart, plot_bar_chart, plot_heatmap
import os
import numpy as np

# Page configuration
st.set_page_config(page_title="District Dashboard", layout="wide")

# Function to load CSS
def load_css(file_name):
    abs_path = os.path.join(os.path.dirname(__file__), "..", file_name)
    if os.path.exists(abs_path):
        with open(abs_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"District CSS file not found: {abs_path}")
load_css("style.css")

# Load Data
@st.cache_data(ttl=3600)
def get_district_data():
    health_df = load_health_data()
    geojson_gdf = load_geojson_data() # This is now a GeoDataFrame
    
    if health_df.empty or geojson_gdf is None or geojson_gdf.empty:
        return None, None, None # Indicate error
        
    merged_gdf = merge_health_data_with_geojson(health_df, geojson_gdf)
    return health_df, geojson_gdf, merged_gdf

health_df_dist, geojson_gdf_dist, merged_gdf_dist = get_district_data()

if health_df_dist is None or merged_gdf_dist is None:
    st.error("Could not load necessary data. District Dashboard cannot be displayed.")
else:
    st.title("🗺️ District Health Officer Dashboard")
    st.markdown("**Strategic Overview for Population Health Management & Resource Allocation**")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("District Filters")
    min_date_dist = health_df_dist['date'].min().date()
    max_date_dist = health_df_dist['date'].max().date()
    
    start_date_dist, end_date_dist = st.sidebar.date_input(
        "Select Date Range for Trends/Analysis",
        [max_date_dist - pd.Timedelta(days=90), max_date_dist], # Default to last 90 days
        min_value=min_date_dist,
        max_value=max_date_dist,
        key="district_date_range_filter"
    )

    # Filter health_df for trends based on date range
    if start_date_dist and end_date_dist and start_date_dist <= end_date_dist:
        filtered_health_df_dist = health_df_dist[
            (health_df_dist['date'].dt.date >= start_date_dist) &
            (health_df_dist['date'].dt.date <= end_date_dist)
        ]
        # Re-merge for current view if map data needs to be date-filtered (optional for some map views)
        # For now, map KPIs use all-time merged_gdf_dist, trends use filtered_health_df_dist
    else:
        st.sidebar.warning("Invalid date range selected for trends. Showing all data.")
        filtered_health_df_dist = health_df_dist.copy()


    # --- KPIs from merged_gdf (represents current state or all-time summary) ---
    district_kpis = get_district_summary_kpis(merged_gdf_dist)
    st.subheader("District-Wide Key Performance Indicators")
    kpi_cols = st.columns(3) # Reduced to 3 for better spacing
    with kpi_cols[0]:
        render_kpi_card("Avg. Population Risk", f"{district_kpis['avg_population_risk']:.1f}", "🎯", 
                        status="High" if district_kpis['avg_population_risk'] > 65 else "Moderate",
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols[1]:
        render_kpi_card("Facility Coverage Score", f"{district_kpis['overall_facility_coverage']:.1f}%", "🏥",
                        status="Low" if district_kpis['overall_facility_coverage'] < 60 else "Moderate",
                        help_text="Population-weighted average facility coverage score (access & capacity).")
    with kpi_cols[2]:
        render_kpi_card("High-Risk Zones", str(district_kpis['zones_high_risk']), "⚠️",
                        status="High" if district_kpis['zones_high_risk'] > 1 else "Moderate",
                        help_text="Number of zones with average risk score > 70.")
    st.markdown("---")

    # --- Choropleth Map ---
    st.subheader("Interactive Health Map: Risk & Resources")
    
    map_metric_options = {
        "Average AI Risk Score": "avg_risk_score",
        "Active Cases (Count)": "active_cases",
        "Prevalence per 1,000": "prevalence_per_1000",
        "Facility Coverage Score": "facility_coverage_score",
        "Population": "population", # From GeoJSON
        "Socio-Economic Index": "socio_economic_index" # From GeoJSON
    }
    selected_map_metric_display = st.selectbox("Select Metric to Display on Map:", list(map_metric_options.keys()), key="map_metric_select")
    selected_map_metric_col = map_metric_options[selected_map_metric_display]

    if selected_map_metric_col in merged_gdf_dist.columns:
        # Determine color scale based on metric (higher is worse for risk/cases, better for coverage)
        color_scale = "OrRd" # Default: Oranges to Reds (higher is worse)
        if "coverage" in selected_map_metric_col.lower() or "socio_economic" in selected_map_metric_col.lower():
            color_scale = "Viridis" # Higher is better (Greenish)

        st.plotly_chart(plot_layered_choropleth_map(
            merged_gdf_dist,
            value_col=selected_map_metric_col,
            title=f"{selected_map_metric_display} by Zone",
            id_col='zone_id',
            featureidkey_prop='zone_id', # Make sure this matches your GeoJSON property used for linking
            color_continuous_scale=color_scale,
            hover_cols=['name', 'population', selected_map_metric_col], # Add more relevant hover info
            # facility_gdf=geojson_gdf_dist, # Can pass the same gdf if it has point geometries for clinics
            # facility_size_col='num_clinics' # Example if you want to size clinic markers
        ), use_container_width=True)
    else:
        st.warning(f"Metric '{selected_map_metric_display}' not available for mapping. Check data processing.")
        
    st.markdown("---")
    
    # --- Tabs for Trends and Comparative Analysis ---
    tab_trends, tab_comparison, tab_correlations = st.tabs(["📈 District Trends", "📊 Zonal Comparison", "🔗 Syndromic Surveillance"])

    with tab_trends:
        st.subheader("Key Health Trends (District-Wide)")
        trend_cols = st.columns(2)
        with trend_cols[0]:
            overall_risk_trend = get_trend_data(filtered_health_df_dist, 'ai_risk_score', period='W') # Weekly trend
            if not overall_risk_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    overall_risk_trend, "Weekly Avg. AI Risk Score (District)", y_axis_title="Avg. Risk Score"
                ), use_container_width=True)
        
        with trend_cols[1]:
            # Example: Trend of new cases (TB and Malaria)
            filtered_health_df_dist['new_case'] = filtered_health_df_dist['condition'].isin(['TB', 'Malaria']).astype(int)
            new_cases_trend = get_trend_data(filtered_health_df_dist, 'new_case', period='W') # Sum of new cases
            new_cases_trend = filtered_health_df_dist.groupby(pd.Grouper(key='date', freq='W'))['new_case'].sum()

            if not new_cases_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    new_cases_trend, "Weekly New Cases (TB & Malaria)", y_axis_title="Number of New Cases"
                ), use_container_width=True)

    with tab_comparison:
        st.subheader("Comparative Analysis Across Zones")
        if not merged_gdf_dist.empty:
            # Select metrics for comparison
            comparison_metrics = ['avg_risk_score', 'prevalence_per_1000', 'facility_coverage_score']
            comparison_df = merged_gdf_dist[['name'] + comparison_metrics].copy()
            comparison_df.set_index('name', inplace=True)
            
            st.dataframe(comparison_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn_r', subset=comparison_metrics, axis=0), use_container_width=True)
            
            # Bar chart for a selected metric
            comp_metric_choice = st.selectbox("Select metric for bar chart comparison:", comparison_metrics, key="comp_metric_bar")
            if comp_metric_choice:
                bar_df = merged_gdf_dist[['name', comp_metric_choice]].sort_values(by=comp_metric_choice, ascending=False)
                st.plotly_chart(plot_bar_chart(
                    bar_df, x_col='name', y_col=comp_metric_choice, 
                    title=f"{comp_metric_choice.replace('_',' ').title()} by Zone",
                    x_axis_title="Zone Name"
                ), use_container_width=True)
        else:
            st.info("No zonal data available for comparison.")
            
    with tab_correlations:
        st.subheader("Syndromic Surveillance (Illustrative)")
        # Mocking syndromic data correlation - in reality, this needs actual symptom data
        # For demonstration, create a mock correlation matrix based on conditions
        if not filtered_health_df_dist.empty:
            conditions = filtered_health_df_dist['condition'].dropna().unique()
            if len(conditions) > 1:
                # Create dummy variables for conditions
                dummy_conditions = pd.get_dummies(filtered_health_df_dist['condition'], prefix='cond')
                # Attempt correlation - this is a very rough example
                # A real syndromic surveillance would correlate actual reported symptoms over time per zone.
                syndromic_corr = dummy_conditions.corr() 
                st.plotly_chart(plot_heatmap(
                    syndromic_corr, "Illustrative Condition Co-occurrence (Correlation)"
                ), use_container_width=True)
                st.caption("Note: This is a simplified correlation based on co-occurrence of diagnosed conditions in the dataset, not true syndromic symptom data.")
            else:
                st.info("Not enough condition diversity for correlation example.")
        else:
            st.info("No data for syndromic correlation example.")

    # Intervention Planner (Placeholder - would require more backend logic)
    st.markdown("---")
    with st.expander("Intervention Planning Tool (Conceptual)", expanded=False):
        st.markdown("""
        This section would allow district officers to:
        - Identify priority zones based on map data and KPIs.
        - Simulate impact of interventions (e.g., deploying mobile clinics, health campaigns).
        - Allocate resources and track progress.
        *This functionality requires significant backend logic and data integration not implemented in this demo.*
        """)
        # Example: simple table of high-risk zones
        high_risk_zones_df = merged_gdf_dist[merged_gdf_dist['avg_risk_score'] >= 70][['name', 'population', 'avg_risk_score', 'facility_coverage_score']].sort_values(by='avg_risk_score', ascending=False)
        if not high_risk_zones_df.empty:
            st.write("Zones identified for potential intervention (Avg. Risk Score >= 70):")
            st.dataframe(high_risk_zones_df, use_container_width=True)
        else:
            st.write("No zones currently meet high-risk criteria for intervention based on risk score >= 70.")
