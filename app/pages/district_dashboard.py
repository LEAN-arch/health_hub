import streamlit as st
import pandas as pd
from utils.data_processor import load_geojson
from utils.viz_helper import plot_line_chart, render_kpi_card, render_traffic_light, plot_heatmap, plot_choropleth_map

# Mock data
mock_data = {
    "disease_trends": [2.2, 2.5, 2.8, 2.7, 2.9, 3.0, 3.2],
    "velocity": {"value": 95, "status": "High", "action": "Boost testing in Zone C"},
    "velocity_trend": [50, 60, 75, 85, 90, 92, 95],
    "anomaly": {"message": "Respiratory Spike", "status": "High", "confidence": 95, "action": "Investigate Zone C"},
    "action_trend": [1.5, 1.4, 1.2, 1.1, 1.0, 1.0, 0.9],
    "equity": {"value": 80, "status": "Moderate"},
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
corr_matrix = pd.DataFrame(
    mock_data['syndromic_correlations'],
    index=["Fever", "Respiratory", "Fatigue"],
    columns=["Fever", "Respiratory", "Fatigue"]
)
st.plotly_chart(plot_heatmap(
    corr_matrix,
    "Syndromic Correlations"
), use_container_width=True)

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