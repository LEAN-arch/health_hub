import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import load_geojson
from utils.viz_helper import plot_annotated_line_chart, render_kpi_card, render_traffic_light, plot_heatmap, plot_layered_choropleth_map
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data
mock_data = {
    "prevalence_rate": 12.5,  # Cases per 1,000
    "disease_trends": [2.2, 2.5, 2.8, 2.7, 2.9, 3.0, 3.2],
    "outbreak_risk": {"value": 85, "ci_lower": 80, "ci_upper": 90, "status": "High", "action": "Scale testing in Zone C"},
    "outbreak_trend": [50, 60, 75, 85, 90, 92, 95],
    "anomaly": {"message": "Respiratory Spike", "status": "High", "confidence": 95, "action": "Investigate Zone C"},
    "action_trend": [1.5, 1.4, 1.2, 1.1, 1.0, 1.0, 0.9],
    "facility_coverage": 70,  # % population within 5km of a facility
    "syndromic_correlations": [
        [1.0, 0.8, 0.3],  # Fever
        [0.8, 1.0, 0.5],  # Respiratory
        [0.3, 0.5, 1.0]   # Fatigue
    ]
}

# Date labels
date_labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"]

# Load GeoJSON
geojson_data = load_geojson("data/zones.geojson")

st.header("District Dashboard")
st.markdown("**Strategic Insights for District Officers**")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("Prevalence Rate", f"{mock_data['prevalence_rate']}/1,000", "📊", status="High")
with col2:
    render_kpi_card("Outbreak Risk", f"{mock_data['outbreak_risk']['value']}% ({mock_data['outbreak_risk']['ci_lower']}-{mock_data['outbreak_risk']['ci_upper']}%)", "⚠️", status=mock_data['outbreak_risk']['status'])
with col3:
    render_kpi_card("Facility Coverage", f"{mock_data['facility_coverage']}%", "🏥", status="Moderate")
with col4:
    render_kpi_card("Action Time", f"{mock_data['action_trend'][-1]} days", "⏱️", status="Low")

# Choropleth Map
st.subheader("AI Disease Risk Map")
if geojson_data:
    try:
        # Validate GeoJSON properties
        for feature in geojson_data["features"]:
            if "name" not in feature["properties"]:
                raise KeyError(f"Missing 'name' property in GeoJSON feature: {feature['properties']}")
            if "risk" not in feature["properties"]:
                raise KeyError(f"Missing 'risk' property in GeoJSON feature: {feature['properties']}")
            if "facilities" not in feature["properties"]:
                raise KeyError(f"Missing 'facilities' property in GeoJSON feature: {feature['properties']}")

        zone_data = pd.DataFrame([
            {"name": f["properties"]["name"], "risk": f["properties"]["risk"], "facilities": f["properties"]["facilities"]}
            for f in geojson_data["features"]
        ])
        if not zone_data.empty:
            st.plotly_chart(plot_layered_choropleth_map(
                geojson_data,
                zone_data,
                "name",
                "risk",
                "facilities",
                "Disease Risk and Health Facilities by Zone"
            ), use_container_width=True)
        else:
            st.warning("No zone data available for choropleth map.")
    except Exception as e:
        st.error(f"Error rendering choropleth map: {str(e)}")
        logger.error(f"Choropleth error: {str(e)}")
else:
    st.warning("GeoJSON data not available. Check 'data/zones.geojson'.")

# Disease Trends
st.subheader("AI Disease Risk Trends")
st.plotly_chart(plot_annotated_line_chart(
    date_labels,
    mock_data['disease_trends'],
    "Risk (per 1,000)",
    "blue",
    target_line=2.5,
    target_label="High: 2.5"
), use_container_width=True)

# Outbreak Risk & Anomalies
st.subheader("Outbreak Risk & Anomalies")
render_traffic_light(f"Outbreak: {mock_data['outbreak_risk']['action']}", mock_data['outbreak_risk']['status'])
render_traffic_light(f"{mock_data['anomaly']['message']} ({mock_data['anomaly']['confidence']}%)", mock_data['anomaly']['status'])
st.plotly_chart(plot_annotated_line_chart(
    date_labels,
    mock_data['outbreak_trend'],
    "Outbreak Risk (%)",
    "red",
    target_line=80,
    target_label="Critical: 80"
), use_container_width=True)

# Syndromic Correlations
st.subheader("Syndromic Correlations")
try:
    corr_matrix = pd.DataFrame(
        mock_data['syndromic_correlations'],
        index=["Fever", "Respiratory", "Fatigue"],
        columns=["Fever", "Respiratory", "Fatigue"]
    ).astype(float)
    if not corr_matrix.empty and corr_matrix.shape[0] == corr_matrix.shape[1]:
        st.plotly_chart(plot_heatmap(
            corr_matrix,
            "Syndromic Correlations"
        ), use_container_width=True)
        logger.info("Successfully rendered syndromic correlations heatmap")
    else:
        st.warning("Invalid correlation matrix: Must be non-empty and square.")
        logger.warning(f"Invalid corr_matrix shape: {corr_matrix.shape}")
except Exception as e:
    st.error(f"Error rendering heatmap: {str(e)}")
    logger.error(f"Heatmap error: {str(e)}")

# Intervention Planner
st.subheader("Intervention Planner")
interventions = [
    {"action": "Deploy mobile testing to Zone C", "priority": "High", "resources": "2 teams, 500 kits"},
    {"action": "Increase facility staff in Zone A", "priority": "Moderate", "resources": "10 nurses"}
]
st.dataframe(pd.DataFrame(interventions), use_container_width=True)

# Export
st.subheader("Export Report")
if st.button("Download Report", key="district_export"):
    df = pd.DataFrame({
        "Metric": ["Prevalence Rate", "Outbreak Risk", "Facility Coverage", "Action Time"],
        "Value": [mock_data['prevalence_rate'], mock_data['outbreak_risk']['value'], mock_data['facility_coverage'], mock_data['action_trend'][-1]],
        "Status": ["High", mock_data['outbreak_risk']['status'], "Moderate", "Low"]
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "district_report.csv", "text/csv")
