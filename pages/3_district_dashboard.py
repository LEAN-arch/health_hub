# pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd # Keep for type hints if needed, though GDF passed in
import os
import logging
from config import app_config
from utils.core_data_processing import (
    # load_health_records, load_zone_data, # These are called within get_district_page_data
    enrich_zone_geodata_with_health_aggregates, # Might be called if get_district_page_data only returns components
    get_district_summary_kpis,
    get_trend_data,
    hash_geodataframe, # Ensure this is imported if used in @st.cache_data for get_district_page_data
    load_health_records, # Actually get_district_page_data calls these, so direct import might not be needed here
    load_zone_data
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_layered_choropleth_map,
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_heatmap # For potential future use
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

# --- Data Loading (copied from your previous complete file, ensure hash_geodataframe is accessible) ---
@st.cache_data(ttl=3600, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None,
    gpd.GeoDataFrame: hash_geodataframe
})
def get_district_page_data():
    logger.info("Getting district page data...")
    health_df = load_health_records()
    zone_base_gdf = load_zone_data()

    if health_df.empty and (zone_base_gdf is None or zone_base_gdf.empty):
        st.error("🚨 CRITICAL ERROR: Both health records and zone geographic data failed to load. Dashboard cannot be displayed.")
        logger.critical("Load_health_records returned empty AND load_zone_data returned None/empty.")
        return pd.DataFrame(columns=['date']), gpd.GeoDataFrame(crs="EPSG:4326") # Return empty with schema

    if health_df.empty:
        logger.warning("Health records are empty. Some dashboard features will be limited or show no data.")
        if zone_base_gdf is not None and not zone_base_gdf.empty:
            # Define expected columns for an empty health_df for enrich function
            expected_health_cols = ['date', 'zone_id', 'patient_id', 'condition', 'ai_risk_score', 'chw_visit', 'test_type']
            empty_health_for_enrich = pd.DataFrame(columns=expected_health_cols)
            enriched_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, empty_health_for_enrich)
            return health_df, enriched_gdf
        else:
             return pd.DataFrame(columns=['date']), gpd.GeoDataFrame(crs="EPSG:4326")

    if zone_base_gdf is None or zone_base_gdf.empty:
        st.error("🚨 ERROR: Could not load zone geographic data. Map and zonal analysis will be unavailable.")
        logger.error("load_zone_data returned None or empty GeoDataFrame.")
        return health_df, gpd.GeoDataFrame(crs="EPSG:4326")

    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(zone_base_gdf, health_df)

    if enriched_zone_gdf is None or enriched_zone_gdf.empty:
         st.warning("⚠️ Warning: Failed to merge health aggregates with zone geographic data. Map/zonal stats may be based on base zone data or be incomplete.")
         logger.warning("Enrichment of zone GDF resulted in empty or None GDF. Falling back to base_zone_gdf if available.")
         return health_df, zone_base_gdf if zone_base_gdf is not None else gpd.GeoDataFrame(crs="EPSG:4326")

    logger.info("Successfully retrieved and processed district page data.")
    return health_df, enriched_zone_gdf

health_records_for_trends, district_map_and_zonal_stats_gdf = get_district_page_data()

# --- Main Page Structure (Title, KPIs, Map - assuming these are as before) ---
st.title("🗺️ District Health Officer Dashboard")
st.markdown("**Strategic Overview for Population Health Management & Resource Allocation in Tijuana**")
st.markdown("---")

# Sidebar Filters (assuming this logic is sound from previous versions)
st.sidebar.header("District Filters")
start_date_filter, end_date_filter = None, None
if not health_records_for_trends.empty and 'date' in health_records_for_trends.columns and pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']):
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
            min_value=min_date, max_value=max_date, key="district_date_filter_final",
            help="This date range applies to time-series trend charts."
        )
    else: st.sidebar.warning("Date range in health records is invalid.")
else: st.sidebar.warning("Health records insufficient for date filter setup.")

filtered_health_records_for_trends = pd.DataFrame()
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not health_records_for_trends.empty:
    filtered_health_records_for_trends = health_records_for_trends[
        (health_records_for_trends['date'].dt.date >= start_date_filter) &
        (health_records_for_trends['date'].dt.date <= end_date_filter)
    ].copy()
elif not health_records_for_trends.empty:
    logger.info("Using all available trend data due to filter issue or no filter set.")
    filtered_health_records_for_trends = health_records_for_trends.copy()


# KPIs and Map (assuming these sections are complete and working from previous versions)
if not district_map_and_zonal_stats_gdf.empty:
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf)
    st.subheader("District-Wide Key Performance Indicators (Overall Aggregates)")
    kpi_cols = st.columns(3); # ... (KPI rendering logic as before) ...
    with kpi_cols[0]: render_kpi_card("Avg. Population Risk", f"{district_kpis.get('avg_population_risk', 0):.1f}", "🎯", status="High" if district_kpis.get('avg_population_risk', 0) > 65 else "Moderate", help_text="Population-weighted average AI risk score.")
    with kpi_cols[1]: render_kpi_card("Facility Coverage Score", f"{district_kpis.get('overall_facility_coverage', 0):.1f}%", "🏥", status="Low" if district_kpis.get('overall_facility_coverage', 0) < 60 else "Moderate", help_text="Population-weighted average facility coverage score.")
    with kpi_cols[2]: render_kpi_card("High-Risk Zones", str(district_kpis.get('zones_high_risk', 0)), "⚠️", status="High" if district_kpis.get('zones_high_risk', 0) > 1 else "Moderate", help_text=f"Zones with avg. risk score >= {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
else: st.warning("KPI data unavailable.")
st.markdown("---")

if not district_map_and_zonal_stats_gdf.empty:
    st.subheader("Interactive Health Map: Tijuana Risk & Resources by Zone")
    # ... (Map selectbox and plotting logic as before, ensuring Tijuana center/zoom is passed) ...
    map_metric_options = {"Average AI Risk Score": "avg_risk_score", "Active Cases (Count)": "active_cases", "Prevalence per 1,000": "prevalence_per_1000", "Facility Coverage Score": "facility_coverage_score", "Population": "population"}
    available_map_metrics = {k: v for k,v in map_metric_options.items() if v in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v].notna().any()}
    if available_map_metrics:
        selected_map_metric_display = st.selectbox("Select Metric for Map:", list(available_map_metrics.keys()), key="dist_map_metric_sel")
        selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)
        if selected_map_metric_col:
            map_fig = plot_layered_choropleth_map(district_map_and_zonal_stats_gdf, selected_map_metric_col, f"{selected_map_metric_display} by Zone", center_lat=app_config.TIJUANA_CENTER_LAT, center_lon=app_config.TIJUANA_CENTER_LON, zoom_level=app_config.TIJUANA_DEFAULT_ZOOM, mapbox_style="open-street-map") # Default to OSM
            st.plotly_chart(map_fig, use_container_width=True)
    else: st.info("No metrics available for map display.")

else: st.warning("Map data unavailable.")
st.markdown("---")


# --- Tabs for Trends, Comparison, and Interventions ---
tab_trends, tab_comparison, tab_interventions = st.tabs([
    "📈 District Trends", "📊 Zonal Comparison", "🛠️ Intervention Insights"
])

# --- 📈 District Trends Tab ---
with tab_trends:
    st.header("📈 District-Wide Health Trends")
    if not filtered_health_records_for_trends.empty and start_date_filter and end_date_filter:
        st.markdown(f"Displaying trends from **{start_date_filter.strftime('%d %b %Y')}** to **{end_date_filter.strftime('%d %b %Y')}**.")
        
        # Trend 1: Overall AI Risk Score
        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            st.subheader("Avg. Patient AI Risk Score")
            overall_risk_trend = get_trend_data(filtered_health_records_for_trends, 'ai_risk_score', period='W', agg_func='mean')
            if not overall_risk_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    overall_risk_trend, "Weekly Avg. AI Risk Score", y_axis_title="Avg. Risk Score",
                    target_line=app_config.TARGET_PATIENT_RISK_SCORE, target_label=f"Target: {app_config.TARGET_PATIENT_RISK_SCORE}",
                    height=app_config.DEFAULT_PLOT_HEIGHT, show_anomalies=True
                ), use_container_width=True)
                avg_current_risk = overall_risk_trend.iloc[-1] if len(overall_risk_trend)>0 else 0
                avg_prev_risk = overall_risk_trend.iloc[-2] if len(overall_risk_trend)>1 else 0
                delta_risk = avg_current_risk - avg_prev_risk if avg_prev_risk > 0 else None
                st.metric(label="Latest Weekly Avg. Risk", value=f"{avg_current_risk:.1f}", delta=f"{delta_risk:.1f}" if delta_risk is not None else None,
                          help="Change from previous week.")
            else: st.caption("No data for risk score trend in selected period.")
        
        # Trend 2: New Cases of Key Conditions
        with trend_col2:
            st.subheader("New Cases (Key Conditions)")
            trend_df_copy = filtered_health_records_for_trends.copy()
            key_conditions = ['TB', 'Malaria', 'ARI', 'Dengue', 'Pneumonia', 'Hypertension', 'Diabetes']
            trend_df_copy['new_case_flag'] = trend_df_copy['condition'].astype(str).isin(key_conditions).astype(int)
            # For new cases, we usually count distinct patients newly identified or tested for these conditions.
            # This mock data doesn't easily distinguish "new", so we sum flags.
            new_cases_trend = get_trend_data(trend_df_copy, 'new_case_flag', period='W', agg_func='sum')

            if not new_cases_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    new_cases_trend, "Weekly New Cases Reported", y_axis_title="Number of New Cases",
                    height=app_config.DEFAULT_PLOT_HEIGHT, show_anomalies=True
                ), use_container_width=True)
                current_cases = new_cases_trend.iloc[-1] if len(new_cases_trend)>0 else 0
                prev_cases = new_cases_trend.iloc[-2] if len(new_cases_trend)>1 else 0
                delta_cases = current_cases - prev_cases
                st.metric(label="Latest Weekly New Cases", value=f"{current_cases:.0f}", delta=f"{delta_cases:.0f}",
                          help="Change from previous week.")
            else: st.caption("No data for new cases trend in selected period.")

        # Trend 3: Test Positivity Rate (Example)
        st.markdown("---")
        st.subheader("Test Positivity Rate Trend (All Tests)")
        if 'test_result' in filtered_health_records_for_trends.columns and 'test_date' in filtered_health_records_for_trends.columns:
            tests_df = filtered_health_records_for_trends.dropna(subset=['test_date', 'test_result']).copy()
            tests_df['test_date'] = pd.to_datetime(tests_df['test_date'])
            tests_df = tests_df[~tests_df['test_result'].isin(['Unknown', 'N/A', 'Pending', 'nan'])] # Conclusive tests

            if not tests_df.empty:
                tests_df['is_positive'] = (tests_df['test_result'] == 'Positive').astype(int)
                weekly_positives = tests_df.groupby(pd.Grouper(key='test_date', freq='W'))['is_positive'].sum()
                weekly_totals = tests_df.groupby(pd.Grouper(key='test_date', freq='W'))['is_positive'].count()
                
                # Calculate positivity rate, handle division by zero
                weekly_positivity_rate = (weekly_positives / weekly_totals * 100).replace([float('inf'), -float('inf')], 0).fillna(0)

                if not weekly_positivity_rate.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        weekly_positivity_rate, "Weekly Test Positivity Rate", y_axis_title="Positivity Rate (%)",
                        height=app_config.DEFAULT_PLOT_HEIGHT, target_line=10, target_label="Alert Threshold: 10%"
                    ), use_container_width=True)
                else: st.caption("Not enough data for test positivity rate trend.")
            else: st.caption("No conclusive test data for positivity rate trend.")
        else: st.caption("Test result or test date columns missing for positivity rate trend.")
    else:
        st.info("Select a valid date range or check data source for trend analysis.")


# --- 📊 Zonal Comparison Tab ---
with tab_comparison:
    st.header("📊 Zonal Comparative Analysis")
    if not district_map_and_zonal_stats_gdf.empty:
        st.markdown("Comparing zones based on overall aggregated health and resource metrics.")
        
        # Define metrics and their desired sorting behavior (True for ascending is better)
        comparison_metrics_config = {
            "Avg. AI Risk Score": {"col": "avg_risk_score", "higher_is_worse": True, "format": "{:.1f}"},
            "Prevalence per 1,000": {"col": "prevalence_per_1000", "higher_is_worse": True, "format": "{:.1f}"},
            "Facility Coverage Score": {"col": "facility_coverage_score", "higher_is_worse": False, "format": "{:.1f}%"},
            "Population": {"col": "population", "format": "{:,.0f}"}, # No worse/better here for sorting in bar
            "Number of Clinics": {"col": "num_clinics", "higher_is_worse": False, "format": "{:.0f}"},
            "Socio-Economic Index": {"col": "socio_economic_index", "higher_is_worse": False, "format": "{:.2f}"}, # Assuming higher SEI is better
            "CHW Visits": {"col": "chw_visits_in_zone", "format": "{:,.0f}"},
            "Total Tests": {"col": "total_tests_conducted", "format": "{:,.0f}"}
        }
        
        available_comp_metrics = {
            disp_name: details for disp_name, details in comparison_metrics_config.items()
            if details["col"] in district_map_and_zonal_stats_gdf.columns and \
               district_map_and_zonal_stats_gdf[details["col"]].notna().any()
        }

        if available_comp_metrics:
            # Data Table
            st.subheader("Zonal Statistics Table")
            display_cols_data = {'name': "Zone Name"} # Start with name
            for disp_name, details in available_comp_metrics.items():
                display_cols_data[details['col']] = disp_name
            
            # Create DataFrame with display names for columns
            comparison_df_display = district_map_and_zonal_stats_gdf[list(display_cols_data.keys())].copy()
            comparison_df_display.rename(columns=display_cols_data, inplace=True)
            comparison_df_display.set_index("Zone Name", inplace=True)

            # Prepare Styler format dictionary
            style_format_dict = {
                disp_name: details["format"] for disp_name, details in available_comp_metrics.items()
                if "format" in details and disp_name in comparison_df_display.columns
            }
            
            # Define columns for highlighting min/max
            highlight_max_cols = [disp_name for disp_name, d in available_comp_metrics.items() if not d.get("higher_is_worse", False) and disp_name in comparison_df_display.columns]
            highlight_min_cols = [disp_name for disp_name, d in available_comp_metrics.items() if d.get("higher_is_worse", True) and disp_name in comparison_df_display.columns] # Default to higher is worse

            st.dataframe(
                comparison_df_display.style.format(style_format_dict)
                .background_gradient(cmap='Greens_r', subset=highlight_max_cols) # Reversed Greens: light green for max (good)
                .background_gradient(cmap='Reds', subset=highlight_min_cols), # Reds: light red for min (good if lower is better)
                use_container_width=True,
                height=min(len(comparison_df_display) * 35 + 40, 400) # Dynamic height up to a max
            )

            # Bar Chart for Selected Metric
            st.subheader("Visual Comparison by Metric")
            selected_comp_metric_display = st.selectbox(
                "Select metric for bar chart comparison:",
                list(available_comp_metrics.keys()),
                key="district_comp_metric_bar_final"
            )
            selected_details = available_comp_metrics.get(selected_comp_metric_display)

            if selected_details:
                selected_col_name = selected_details["col"]
                # Sort by value: if higher is worse, sort descending for bar chart to show worst first
                sort_ascending_for_bar = selected_details.get("higher_is_worse", True) == False 

                bar_df_comp = district_map_and_zonal_stats_gdf[['name', selected_col_name]].copy()
                
                st.plotly_chart(plot_bar_chart(
                    bar_df_comp, x_col='name', y_col=selected_col_name,
                    title=f"{selected_comp_metric_display} by Zone", x_axis_title="Zone Name",
                    height=app_config.DEFAULT_PLOT_HEIGHT + 100, # Taller for zone names
                    sort_values_by=selected_col_name, ascending=sort_ascending_for_bar,
                    text_auto = '.2s' if "population" in selected_col_name.lower() or "visits" in selected_col_name.lower() or "tests" in selected_col_name.lower() else True
                ), use_container_width=True)
        else:
            st.info("No metrics with valid data available for Zonal Comparison.")
    else:
        st.info("No zonal data available for comparison. Check data loading.")

# --- 🛠️ Intervention Insights Tab ---
with tab_interventions:
    st.header("🛠️ Intervention Planning Insights")
    if not district_map_and_zonal_stats_gdf.empty:
        st.markdown("""
        This section helps identify zones requiring attention based on configurable risk factors.
        The data reflects **overall aggregates** from all available historical records for each zone.
        """)

        # Multi-select for intervention criteria
        st.subheader("Select Intervention Criteria")
        criteria_options = {
            f"High Avg. Risk Score (>= {app_config.RISK_THRESHOLDS['district_zone_high_risk']})": "high_risk",
            "Low Facility Coverage (< 50%)": "low_coverage",
            "High Prevalence (Top 25%)": "high_prevalence"
        }
        selected_criteria_keys = st.multiselect(
            "Filter zones by (meets ANY selected):",
            options=list(criteria_options.keys()),
            default=list(criteria_options.keys())[0:1] # Default to first criterion
        )

        # Build mask based on selected criteria
        priority_masks = []
        if not selected_criteria_keys:
            st.info("Please select at least one criterion to identify priority zones.")
        else:
            for key in selected_criteria_keys:
                criterion = criteria_options[key]
                if criterion == "high_risk" and 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns:
                    priority_masks.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
                elif criterion == "low_coverage" and 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns:
                    priority_masks.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < 50)
                elif criterion == "high_prevalence" and 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns and not district_map_and_zonal_stats_gdf['prevalence_per_1000'].empty:
                    prevalence_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
                    if pd.notna(prevalence_q75) and prevalence_q75 >= 0: # Allow 0 if that's the q75
                        priority_masks.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prevalence_q75)

            if priority_masks:
                try:
                    # Combine masks: a zone is priority if it meets ANY selected criterion (OR logic)
                    final_priority_mask = pd.DataFrame(priority_masks).transpose().any(axis=1)
                    # Align index with the GDF before applying mask
                    final_priority_mask.index = district_map_and_zonal_stats_gdf.index 
                    priority_zones_df = district_map_and_zonal_stats_gdf[final_priority_mask]
                except Exception as e_mask:
                    logger.error(f"Error applying intervention criteria masks: {e_mask}", exc_info=True)
                    priority_zones_df = pd.DataFrame()
                
                if not priority_zones_df.empty:
                    st.markdown("###### Zones Identified for Potential Intervention:")
                    intervention_cols_to_show = ['name', 'population', 'avg_risk_score', 'prevalence_per_1000', 'facility_coverage_score', 'num_clinics']
                    actual_intervention_cols = [col for col in intervention_cols_to_show if col in priority_zones_df.columns]
                    
                    # Sort by multiple factors for better prioritization
                    # Example: highest risk, then lowest coverage, then highest prevalence
                    sort_by_cols = []
                    if 'avg_risk_score' in actual_intervention_cols: sort_by_cols.append('avg_risk_score')
                    if 'facility_coverage_score' in actual_intervention_cols: sort_by_cols.append('facility_coverage_score')
                    if 'prevalence_per_1000' in actual_intervention_cols: sort_by_cols.append('prevalence_per_1000')
                    
                    ascending_order = [False, True, False] # Risk (High->Low), Coverage (Low->High), Prevalence (High->Low)
                    # Filter ascending_order to match existing sort_by_cols
                    actual_ascending_order = [asc for col, asc in zip(['avg_risk_score', 'facility_coverage_score', 'prevalence_per_1000'], ascending_order) if col in sort_by_cols]


                    if sort_by_cols:
                         priority_zones_df_sorted = priority_zones_df.sort_values(by=sort_by_cols, ascending=actual_ascending_order)
                    else:
                         priority_zones_df_sorted = priority_zones_df # No valid sort columns

                    st.dataframe(
                        priority_zones_df_sorted[actual_intervention_cols],
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
                    st.success("✅ No zones currently meet the selected high-priority criteria.")
            elif selected_criteria_keys : # User selected criteria, but they couldn't be applied (e.g. column missing)
                st.warning("Could not apply selected criteria. Check if the required data columns (e.g., 'avg_risk_score', 'facility_coverage_score', 'prevalence_per_1000') are available in the zone data.")
            # else: user selected no criteria, handled by the if not selected_criteria_keys above
    else:
        st.info("No zonal data available for intervention insights. Please check data loading.")
