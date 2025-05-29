import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import load_geojson
from utils.viz_helper import plot_annotated_line_chart, render_kpi_card, render_traffic_light, plot_heatmap, plot_layered_choropleth_map

# Mock data
mock_data = {
    "prevalence_rate": 12.5,  # Cases per 1,000
    "disease_trends": [2.2, 2.5, 2.8, 2.7, 2.9, 3.0, 3.2],
    "outbreak_risk": {"value": 85, "ci_lower": 80, "ci_upper": 90, "status": "High", "action": "Scale testing in Zone C"},
    "outbreak_trend": [50, 60, 75, 85, 90, 92, 95],
    "anomaly": {"message": "Respiratory Spike", "status": "High", "confidence": 95, "action": "Investigate Zone C"},
    "action_trend": [1.5, 1.4, 1.2, 1.1, 1.0, 1.0, 0.9],
    "facility_coverage": 70,  # % population within 5km of a facility
    "syndromic_correlations": {
        "fever": [1.0, 0.8, 0.3],
        "respiratory": [0.8, 1.0, 0.5],
        "fatigue": [0.3, 0.5, 1.0]
    }
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
    zone_data = pd.DataFrame([
        {"zone": f["properties"]["zone"], "risk": f["properties"]["risk"], "facilities": f["properties"]["facilities"]}
        for f in geojson_data["features"]
    ])
    st.plotly_chart(plot_layered_choropleth_map(
        geojson_data,
        zone_data,
        "zone",
        "risk",
        "facilities",
        "Disease Risk and Facilities by Zone"
    ), use_container_width=True)
else:
    st.warning("GeoJSON data not available.")

# Disease Trends
st.subheader("AI Disease Risk Trends")
st.plotly_chart(plot_annotated_line_chart(
    date_labels,
    mock_data['disease_trends'],
    "Risk (per 1,000)",
    "#3b82f6",
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
    "#ef4444",
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
    st.plotly_chart(plot_heatmap(
        corr_matrix,
        "Syndromic Correlations"
    ), use_container_width=True)
except Exception as e:
    st.error(f"Error rendering heatmap: {str(e)}")

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
date_labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"]

# Load GeoJSON
geojson_data = load_geojson("data/zones.geojson")

st.header("District Dashboard")
st.markdown("**Strategic insights for District Officers**")

# KPIs
col1, col2 = st.columns(2)
with col1:
    render_kpi_card("Syndromic Velocity", f"{mock_data['velocity']['value']}%", "📈", status=mock_data['velocity']['status'])
with col2:
    render_kpi_card("Equity Index", f"{mock_data['equity']['value']}%", "⚖️", status=mock_data['equity']['status'])

# Choropleth Map
st.subheader("AI Disease Risk Map")
if geojson_data:
    zone_risks = pd.DataFrame([
        {"zone": f["properties"]["zone"], "risk": f["properties"]["risk"]}
        for f in geojson_data["features"]
    ])
    st.plotly_chart(plot_choropleth_map(
        geojson_data,
        zone_risks,
        "zone",
        "risk",
        "Disease Risk by Zone"
    ), use_container_width=True)
else:
    st.warning("GeoJSON data not available.")

# Disease Trends
st.subheader("AI Disease Risk Trends")
st.plotly_chart(plot_line_chart(
    date_labels,
    mock_data['disease_trends'],
    "Risk (per 1000)",
    "#3b82f6",
    target_line=2.5,
    target_label="High: 2.5"
), use_container_width=True)

# Velocity & Anomalies
st.subheader("Syndromic Velocity & Anomalies")
render_traffic_light(f"Velocity: {mock_data['velocity']['action']}", mock_data['velocity']['status'])
render_traffic_light(f"{mock_data['anomaly']['message']} ({mock_data['anomaly']['confidence']}%)", mock_data['anomaly']['status'])
st.plotly_chart(plot_line_chart(
    date_labels,
    mock_data['velocity_trend'],
    "Velocity (%)",
    "#ef4444",
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
    ).astype(float)  # Ensure numeric type
    st.plotly_chart(plot_heatmap(
        corr_matrix,
        "Syndromic Correlations"
    ), use_container_width=True)
except Exception as e:
    st.error(f"Error rendering heatmap: {str(e)}")

# Action Time
st.subheader("Program Impact")
st.plotly_chart(plot_line_chart(
    date_labels,
    mock_data['action_trend'],
    "Action Time (days)",
    "#22c55e",
    target_line=1.2,
    target_label="Target: 1.2 days"
), use_container_width=True)

# Export
st.subheader("Export Data")
if st.button("Download CSV", key="district_export"):
    df = pd.DataFrame({
        "Metric": ["Velocity", "Equity"],
        "Value": [mock_data['velocity']['value'], mock_data['equity']['value']],
        "Status": [mock_data['velocity']['status'], mock_data['equity']['status']]
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "district_export.csv", "text/csv")
