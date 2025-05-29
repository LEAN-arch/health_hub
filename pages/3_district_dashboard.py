# pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import logging
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_zone_data, load_iot_clinic_environment_data, # Load all three
    enrich_zone_geodata_with_health_aggregates,
    get_district_summary_kpis, get_trend_data, hash_geodataframe
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_layered_choropleth_map, plot_annotated_line_chart,
    plot_bar_chart, plot_heatmap
)

st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

def load_css(): # pragma: no cover
    if os.path.exists(app_config.STYLE_CSS):
        with open(app_config.STYLE_CSS) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"CSS file not found at {app_config.STYLE_CSS}.")
load_css()

@st.cache_data(ttl=3600, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None,
    gpd.GeoDataFrame: hash_geodataframe
})
def get_district_page_data():
    logger.info("Getting district page data (including IoT)...")
    health_df = load_health_records()
    zone_base_gdf = load_zone_data()
    iot_df = load_iot_clinic_environment_data()

    # Handle empty/None data sources early
    health_df_safe = health_df if health_df is not None else pd.DataFrame(columns=['date']) # Provide default columns for enrich
    zone_base_gdf_safe = zone_base_gdf if zone_base_gdf is not None else gpd.GeoDataFrame(crs="EPSG:4326")
    iot_df_safe = iot_df if iot_df is not None else pd.DataFrame()


    if health_df_safe.empty and zone_base_gdf_safe.empty: # pragma: no cover
        st.error("🚨 CRITICAL ERROR: Health records and zone geographic data failed to load.")
        return pd.DataFrame(columns=['date']), gpd.GeoDataFrame(crs="EPSG:4326"), pd.DataFrame()
    
    if health_df_safe.empty: logger.warning("Health records empty for district page.")
    if zone_base_gdf_safe.empty: st.error("🚨 ERROR: Zone geographic data failed to load. Map/zonal analysis unavailable.")
    if iot_df_safe.empty: logger.info("IoT data not available or empty for district page.")


    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_base_gdf_safe, 
        health_df_safe, 
        iot_df_safe # Pass IoT data for enrichment
    )

    if enriched_zone_gdf is None or enriched_zone_gdf.empty: # pragma: no cover
         st.warning("⚠️ Warning: Failed to create enriched zone data. Map/zonal stats may be based on base zone data or be incomplete.")
         return health_df_safe, zone_base_gdf_safe, iot_df_safe

    logger.info("Successfully retrieved and processed district page data including IoT.")
    return health_df_safe, enriched_zone_gdf, iot_df_safe

health_records_for_trends, district_map_and_zonal_stats_gdf, iot_records_for_trends = get_district_page_data()

st.title("🗺️ District Health Officer Dashboard")
st.markdown("**Strategic Overview for Population Health (Tijuana Focus: TB, Malaria, HIV, STIs, NTDs etc.) & Environmental Health**")
st.markdown("---")

# --- Sidebar Filters (as before, ensure robustness) ---
st.sidebar.header("District Filters")
start_date_filter, end_date_filter = None, None
# ... (Robust date filter logic from previous complete file) ...
if not health_records_for_trends.empty and 'date' in health_records_for_trends.columns and pd.api.types.is_datetime64_any_dtype(health_records_for_trends['date']):
    min_date_dt = health_records_for_trends['date'].min(); max_date_dt = health_records_for_trends['date'].max()
    if pd.notna(min_date_dt) and pd.notna(max_date_dt):
        min_date = min_date_dt.date(); max_date = max_date_dt.date()
        default_start_dt = max_date_dt - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
        default_start_date = default_start_dt.date()
        if default_start_date < min_date: default_start_date = min_date
        if default_start_date > max_date and min_date <= max_date : default_start_date = min_date 
        elif default_start_date > max_date : default_start_date = max_date
        start_date_filter, end_date_filter = st.sidebar.date_input("Select Date Range for Trends:", value=[default_start_date, max_date], min_value=min_date, max_value=max_date, key="district_date_filter_final_v2")
    else: st.sidebar.warning("Date range in health records is invalid for filter.") # pragma: no cover
else: st.sidebar.warning("Health records insufficient for date filter setup.") # pragma: no cover
filtered_health_records_for_trends = pd.DataFrame()
if start_date_filter and end_date_filter and start_date_filter <= end_date_filter and not health_records_for_trends.empty:
    filtered_health_records_for_trends = health_records_for_trends[(health_records_for_trends['date'].dt.date >= start_date_filter) & (health_records_for_trends['date'].dt.date <= end_date_filter)].copy()
elif not health_records_for_trends.empty:
    logger.info("Using all available trend data due to filter issue or no filter set."); filtered_health_records_for_trends = health_records_for_trends.copy()

# --- KPIs ---
if not district_map_and_zonal_stats_gdf.empty:
    district_kpis = get_district_summary_kpis(district_map_and_zonal_stats_gdf)
    st.subheader("District-Wide Key Performance Indicators (Overall Aggregates)")
    kpi_cols_1 = st.columns(3)
    with kpi_cols_1[0]: render_kpi_card("Avg. Population Risk", f"{district_kpis.get('avg_population_risk', 0):.1f}", "🎯", status="High" if district_kpis.get('avg_population_risk', 0) > 65 else "Moderate")
    with kpi_cols_1[1]: render_kpi_card("Facility Coverage Score", f"{district_kpis.get('overall_facility_coverage', 0):.1f}%", "🏥", status="Low" if district_kpis.get('overall_facility_coverage', 0) < 60 else "Moderate")
    with kpi_cols_1[2]: render_kpi_card("High-Risk Zones", str(district_kpis.get('zones_high_risk', 0)), "⚠️", status="High" if district_kpis.get('zones_high_risk', 0) > 1 else "Moderate")
    
    st.markdown("##### Key Disease Burden")
    kpi_cols_2 = st.columns(3) # Example: TB, Malaria, HIV
    with kpi_cols_2[0]: render_kpi_card("Active TB Cases", str(district_kpis.get('district_tb_burden',0)), "<img src='https://www.svgrepo.com/show/309948/lungs.svg' width='24'>", status="High" if district_kpis.get('district_tb_burden',0) > 20 else "Moderate") # Threshold example
    with kpi_cols_2[1]: render_kpi_card("Active Malaria Cases", str(district_kpis.get('district_malaria_burden',0)), "<img src='https://www.svgrepo.com/show/491020/mosquito.svg' width='24'>", status="High" if district_kpis.get('district_malaria_burden',0) > 50 else "Moderate")
    # Add HIV KPI if data available: hiv_positive_cases_total = district_map_and_zonal_stats_gdf['hiv_positive_cases'].sum()
    # For Anemia, you might show % of screened women with Anemia if that data is aggregated.
    with kpi_cols_2[2]: 
        avg_co2_iot = district_map_and_zonal_stats_gdf['zone_avg_co2'].mean() if 'zone_avg_co2' in district_map_and_zonal_stats_gdf else 0
        render_kpi_card("Avg. Clinic CO2", f"{avg_co2_iot:.0f} ppm", "💨", status="High" if avg_co2_iot > app_config.CO2_LEVEL_ALERT_PPM else "Moderate", help_text="Average CO2 across clinics in zones.")

else: st.warning("Zone map data for KPIs is unavailable.") # pragma: no cover
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
        "Avg. Patient Steps": "avg_daily_steps_zone", "Avg. Patient SpO2": "avg_spo2_zone",
        "Avg. Zone CO2 (Clinics)": "zone_avg_co2"
    }
    available_map_metrics = {k: v for k, v in map_metric_options.items() if v in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v].notna().any()}
    
    if available_map_metrics:
        selected_map_metric_display = st.selectbox("Select Metric for Map:", list(available_map_metrics.keys()), key="district_map_metric_final")
        selected_map_metric_col = available_map_metrics.get(selected_map_metric_display)
        if selected_map_metric_col:
            color_scale = "OrRd" # Higher is worse default
            if any(keyword in selected_map_metric_col.lower() for keyword in ["coverage", "steps", "spo2", "socio_economic", "clinics"]): color_scale = "Mint" # Higher is better
            
            map_figure = plot_layered_choropleth_map(
                district_map_and_zonal_stats_gdf, value_col=selected_map_metric_col,
                title=f"{selected_map_metric_display} by Zone",
                id_col='zone_id', featureidkey_prop='zone_id', color_continuous_scale=color_scale,
                hover_cols=['name', 'population', selected_map_metric_col, 'num_clinics'], # Simplified hover
                height=app_config.MAP_PLOT_HEIGHT, center_lat=app_config.TIJUANA_CENTER_LAT,
                center_lon=app_config.TIJUANA_CENTER_LON, zoom_level=app_config.TIJUANA_DEFAULT_ZOOM,
                mapbox_style="open-street-map" # Safest default
            )
            st.plotly_chart(map_figure, use_container_width=True)
    else: st.warning("No metrics available for map display.") # pragma: no cover
else: st.warning("Map data unavailable.") # pragma: no cover
st.markdown("---")

# --- Tabs (Trends, Comparison, Interventions) ---
# ... (Tabs logic as in previous complete file, but ensure the metrics in comparison_metrics_config
#      and intervention_criteria match the newly aggregated disease-specific and sensor columns
#      from enrich_zone_geodata_with_health_aggregates) ...

tab_trends, tab_comparison, tab_interventions = st.tabs(["📈 District Trends", "📊 Zonal Comparison", "🛠️ Intervention Insights"])

with tab_trends:
    # ... (Trend logic - ensure it uses filtered_health_records_for_trends) ...
    # ... (Include trends for key diseases: new TB cases, Malaria incidence, HIV testing rates etc.) ...
    # ... (Also include district-level IoT trends from iot_records_for_trends) ...
    if not filtered_health_records_for_trends.empty and start_date_filter and end_date_filter:
        st.subheader(f"Key Health Trends ({start_date_filter.strftime('%d %b %Y')} - {end_date_filter.strftime('%d %b %Y')})")
        trend_cols_main = st.columns(2)
        with trend_cols_main[0]:
            overall_risk_trend = get_trend_data(filtered_health_records_for_trends, 'ai_risk_score', period='W', agg_func='mean')
            if not overall_risk_trend.empty: st.plotly_chart(plot_annotated_line_chart(overall_risk_trend, "Weekly Avg. AI Risk Score", y_axis_title="Avg. Risk Score", target_line=app_config.TARGET_PATIENT_RISK_SCORE), use_container_width=True)
            else: st.caption("No risk score trend.")
        with trend_cols_main[1]:
            trend_df_copy = filtered_health_records_for_trends.copy()
            trend_df_copy['key_condition_flag'] = trend_df_copy['condition'].astype(str).isin(app_config.KEY_CONDITIONS_FOR_TRENDS).astype(int)
            new_key_cond_trend = get_trend_data(trend_df_copy, 'key_condition_flag', period='W', agg_func='sum')
            if not new_key_cond_trend.empty: st.plotly_chart(plot_annotated_line_chart(new_key_cond_trend, "Weekly New Cases (Key Conditions)", y_axis_title="New Cases"), use_container_width=True)
            else: st.caption("No new key condition cases trend.")
        
        if not iot_records_for_trends.empty :
            st.markdown("---"); st.subheader(f"District Environmental Trends (Clinics)")
            filtered_iot_district_trends = iot_records_for_trends[(iot_records_for_trends['timestamp'].dt.date >= start_date_filter) & (iot_records_for_trends['timestamp'].dt.date <= end_date_filter)].copy() if start_date_filter and end_date_filter else iot_records_for_trends.copy()
            iot_trend_cols_dist = st.columns(2)
            with iot_trend_cols_dist[0]:
                district_avg_co2_trend = get_trend_data(filtered_iot_district_trends, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not district_avg_co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(district_avg_co2_trend, "Daily Avg. CO2 (All Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM), use_container_width=True)
                else: st.caption("No CO2 trend.")
            with iot_trend_cols_dist[1]:
                district_avg_steps_trend = get_trend_data(filtered_health_records_for_trends, 'avg_daily_steps', period='W', agg_func='mean')
                if not district_avg_steps_trend.empty: st.plotly_chart(plot_annotated_line_chart(district_avg_steps_trend, "Weekly Avg. Patient Steps", y_axis_title="Avg. Steps", target_line=app_config.TARGET_DAILY_STEPS), use_container_width=True)
                else: st.caption("No patient steps trend.")
    else: st.info("No data for trend analysis for selected period.")


with tab_comparison:
    # ... (Zonal comparison logic from previous "complete fixed code file" - ensure column names match new aggregates)
    if not district_map_and_zonal_stats_gdf.empty:
        st.subheader("Zonal Comparative Analysis (Overall Aggregates)")
        comparison_metrics_config_dist = {
            "Avg. AI Risk Score": {"col": "avg_risk_score", "higher_is_worse": True, "format": "{:.1f}"},
            "Active TB Cases": {"col": "active_tb_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "Active Malaria Cases": {"col": "active_malaria_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "HIV Positive Cases": {"col": "hiv_positive_cases", "higher_is_worse": True, "format": "{:.0f}"},
            "Facility Coverage": {"col": "facility_coverage_score", "higher_is_worse": False, "format": "{:.1f}%"},
            "Avg. Patient Steps": {"col": "avg_daily_steps_zone", "higher_is_worse": False, "format":"{:,.0f}"},
            "Avg. Zone CO2": {"col": "zone_avg_co2", "higher_is_worse": True, "format": "{:.0f} ppm"}
        }
        available_comp_metrics_dist = {k: v for k,v in comparison_metrics_config_dist.items() if v["col"] in district_map_and_zonal_stats_gdf.columns and district_map_and_zonal_stats_gdf[v["col"]].notna().any()}
        if available_comp_metrics_dist:
            # ... (dataframe display and bar chart selectbox logic as before, using available_comp_metrics_dist) ...
            display_comp_cols_dist = ['name'] + [d['col'] for d in available_comp_metrics_dist.values() if d['col'] != 'name']; display_comp_cols_dist = list(dict.fromkeys(display_comp_cols_dist))
            comparison_df_display_dist = district_map_and_zonal_stats_gdf[display_comp_cols_dist].copy(); comparison_df_display_dist.rename(columns={'name':'Zone Name'}, inplace=True); comparison_df_display_dist.set_index('Zone Name', inplace=True)
            style_format_dict_dist = {available_comp_metrics_dist[disp_name]["col"]: details["format"] for disp_name, details in available_comp_metrics_dist.items() if "format" in details and available_comp_metrics_dist[disp_name]["col"] in comparison_df_display_dist.columns} # Map to actual col names
            # Remap display names for subset highlighting
            highlight_max_cols_dist = [available_comp_metrics_dist[disp_name]["col"] for disp_name, d in available_comp_metrics_dist.items() if not d.get("higher_is_worse", False) and available_comp_metrics_dist[disp_name]["col"] in comparison_df_display_dist.columns]
            highlight_min_cols_dist = [available_comp_metrics_dist[disp_name]["col"] for disp_name, d in available_comp_metrics_dist.items() if d.get("higher_is_worse", True) and available_comp_metrics_dist[disp_name]["col"] in comparison_df_display_dist.columns]
            st.dataframe(comparison_df_display_dist.style.format(style_format_dict_dist).background_gradient(cmap='Greens_r', subset=highlight_max_cols_dist).background_gradient(cmap='Reds', subset=highlight_min_cols_dist),use_container_width=True, height=min(len(comparison_df_display_dist) * 35 + 40, 400))
            selected_comp_metric_display_dist = st.selectbox("Select metric for bar chart:", list(available_comp_metrics_dist.keys()), key="district_comp_bar_final_v2")
            selected_details_dist = available_comp_metrics_dist.get(selected_comp_metric_display_dist)
            if selected_details_dist:
                selected_col_name_dist = selected_details_dist["col"]; sort_ascending_bar_dist = selected_details_dist.get("higher_is_worse", True) == False
                bar_df_comp_dist = district_map_and_zonal_stats_gdf[['name', selected_col_name_dist]].copy()
                st.plotly_chart(plot_bar_chart(bar_df_comp_dist, x_col='name', y_col=selected_col_name_dist, title=f"{selected_comp_metric_display_dist} by Zone", x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 70, sort_values_by=selected_col_name_dist, ascending=sort_ascending_bar_dist, text_auto = '.2s' if "population" in selected_col_name_dist.lower() or "visits" in selected_col_name_dist.lower() or "tests" in selected_col_name_dist.lower() else True), use_container_width=True)
        else: st.info("No metrics available for Zonal Comparison.") # pragma: no cover
    else: st.info("No zonal data for comparison.") # pragma: no cover


with tab_interventions:
    # ... (Intervention logic from previous complete file, ensure criteria options include new disease/sensor metrics) ...
    if not district_map_and_zonal_stats_gdf.empty:
        st.subheader("Intervention Planning Insights"); st.markdown("Identify zones based on combined risk factors. Data shown aggregates all available historical records.")
        criteria_options_int = {
            f"High Avg. Risk Score (>= {app_config.RISK_THRESHOLDS['district_zone_high_risk']})": "high_risk",
            "Low Facility Coverage (< 50%)": "low_coverage",
            "High Prevalence (Key Inf., Top 25%)": "high_prevalence", # Uses total_active_key_infections
            "High TB Burden (e.g., >5 cases)": "high_tb_burden", # Define specific threshold
            "High Avg. Clinic CO2 (>900ppm)": "high_clinic_co2"
        }
        selected_criteria_keys_int = st.multiselect("Filter zones by (meets ANY selected):", options=list(criteria_options_int.keys()), default=list(criteria_options_int.keys())[0:1], key="intervention_criteria_final")
        priority_masks_int = []
        if not selected_criteria_keys_int: st.info("Please select at least one criterion to identify priority zones.")
        else:
            for key_int in selected_criteria_keys_int:
                criterion_int = criteria_options_int[key_int]
                if criterion_int == "high_risk" and 'avg_risk_score' in district_map_and_zonal_stats_gdf.columns: priority_masks_int.append(district_map_and_zonal_stats_gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk'])
                elif criterion_int == "low_coverage" and 'facility_coverage_score' in district_map_and_zonal_stats_gdf.columns: priority_masks_int.append(district_map_and_zonal_stats_gdf['facility_coverage_score'] < 50)
                elif criterion_int == "high_prevalence" and 'prevalence_per_1000' in district_map_and_zonal_stats_gdf.columns and not district_map_and_zonal_stats_gdf['prevalence_per_1000'].empty :
                    prev_q75 = district_map_and_zonal_stats_gdf['prevalence_per_1000'].quantile(0.75)
                    if pd.notna(prev_q75) and prev_q75 >= 0: priority_masks_int.append(district_map_and_zonal_stats_gdf['prevalence_per_1000'] >= prev_q75)
                elif criterion_int == "high_tb_burden" and 'active_tb_cases' in district_map_and_zonal_stats_gdf.columns: priority_masks_int.append(district_map_and_zonal_stats_gdf['active_tb_cases'] > 5) # Example: > 5 cases
                elif criterion_int == "high_clinic_co2" and 'zone_avg_co2' in district_map_and_zonal_stats_gdf.columns: priority_masks_int.append(district_map_and_zonal_stats_gdf['zone_avg_co2'] > 900) # Example: > 900ppm
            if priority_masks_int:
                try:
                    aligned_criteria_int = [s.reindex(district_map_and_zonal_stats_gdf.index).fillna(False) for s in priority_masks_int if isinstance(s, pd.Series)]
                    if aligned_criteria_int: final_priority_mask_int = pd.concat(aligned_criteria_int, axis=1).any(axis=1)
                    else: final_priority_mask_int = pd.Series([False]*len(district_map_and_zonal_stats_gdf), index=district_map_and_zonal_stats_gdf.index)
                    priority_zones_df_int = district_map_and_zonal_stats_gdf[final_priority_mask_int]
                except Exception as e_mask_int: logger.error(f"Error applying intervention criteria masks: {e_mask_int}", exc_info=True); priority_zones_df_int = pd.DataFrame()
                if not priority_zones_df_int.empty:
                    st.markdown("###### Zones Identified for Potential Intervention:"); 
                    # ... (dataframe display logic as before) ...
                    intervention_cols_show = ['name', 'population', 'avg_risk_score', 'active_tb_cases', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2']
                    actual_int_cols = [col for col in intervention_cols_show if col in priority_zones_df_int.columns]
                    sort_by_cols_int = [col for col in ['avg_risk_score', 'active_tb_cases', 'zone_avg_co2'] if col in actual_int_cols] # Example sort
                    ascending_int = [False, False, False] # Higher risk, more TB, higher CO2 = higher priority
                    priority_zones_df_int_sorted = priority_zones_df_int.sort_values(by=sort_by_cols_int, ascending=ascending_int) if sort_by_cols_int else priority_zones_df_int
                    st.dataframe(priority_zones_df_int_sorted[actual_int_cols], use_container_width=True, column_config={"population": st.column_config.NumberColumn(format="%d"), "avg_risk_score": st.column_config.NumberColumn(format="%.1f"), "active_tb_cases":st.column_config.NumberColumn(format="%d"), "prevalence_per_1000": st.column_config.NumberColumn(format="%.1f"), "facility_coverage_score": st.column_config.NumberColumn(format="%.1f%%"), "zone_avg_co2":st.column_config.NumberColumn(format="%.0f ppm")})
                else: st.success("✅ No zones meet selected high-priority criteria.")
            elif selected_criteria_keys_int: st.warning("Could not apply selected criteria. Check data columns.")
    else: st.info("No zonal data for intervention insights.") # pragma: no cover
