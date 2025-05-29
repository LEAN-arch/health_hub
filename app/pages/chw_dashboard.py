import streamlit as st
import pandas as pd
from utils.data_processor import load_data
from utils.viz_helper import plot_line_chart, render_kpi_card, render_traffic_light, plot_bar_chart

# Mock data
mock_data = {
    "referrals": 5,
    "referrals_trend": [3, 4, 5, 4, 6, 5, 5],
    "badge_adherence": 85,
    "badge_trend": [80, 82, 84, 83, 85, 84, 85],
    "symptom_risk": {"message": "Fever Cluster", "risk": 75, "status": "Moderate", "action": "Escalate triage"},
    "symptom_trend": [50, 55, 60, 65, 70, 72, 75],
    "alerts": [
        {"id": "P001", "issue": "Acute BP Spike", "status": "High", "type": "BP"},
        {"id": "P002", "issue": "TB Risk", "status": "Moderate", "type": "TB"},
        {"id": "P003", "issue": "TB Risk", "status": "Moderate", "type": "TB"}
    ],
    "risk_zones": [{"zone": "Zone A", "issue": "High CO₂", "status": "High"}]
}

# Date labels
date_labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"]

st.header("CHW Dashboard")
st.markdown("**Real-time field insights for Community Health Workers**")

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    render_kpi_card("Pending Referrals", f"{mock_data['referrals']} patients", "📋")
with col2:
    render_kpi_card("Badge Adherence", f"{mock_data['badge_adherence']}%", "🛡️")
with col3:
    render_kpi_card("AI Symptom Risk", f"{mock_data['symptom_risk']['risk']}%", "⚠️", status=mock_data['symptom_risk']['status'])

# Alerts
st.subheader("Critical Alerts")
for alert in mock_data['alerts']:
    render_traffic_light(f"{alert['id']}: {alert['issue']}", alert['status'])

# Alert Breakdown
st.subheader("Alert Breakdown by Type")
alert_types = pd.DataFrame(mock_data['alerts']).groupby("type").size().reset_index(name="count")
st.plotly_chart(plot_bar_chart(
    alert_types["type"],
    alert_types["count"],
    "Alert Counts by Type",
    "#3b82f6"
), use_container_width=True)

# Risk Zones
st.subheader("High-Risk Zones")
for zone in mock_data['risk_zones']:
    render_traffic_light(f"{zone['zone']}: {zone['issue']}", zone['status'])

# Trends
st.subheader("Trends")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_line_chart(
        date_labels,
        mock_data['referrals_trend'],
        "Referrals Trend",
        "#3b82f6",
        target_line=5,
        target_label="Target: 5"
    ), use_container_width=True)
with col2:
    st.plotly_chart(plot_line_chart(
        date_labels,
        mock_data['badge_trend'],
        "Badge Adherence (%)",
        "#22c55e",
        target_line=90,
        target_label="Target: 90%"
    ), use_container_width=True)

# AI Symptom Prediction
st.subheader("AI Symptom Prediction")
st.markdown(f"**{mock_data['symptom_risk']['message']}**: {mock_data['symptom_risk']['action']}")
st.plotly_chart(plot_line_chart(
    date_labels,
    mock_data['symptom_trend'],
    "AI Risk Score (%)",
    "#facc15",
    target_line=70,
    target_label="Alert: 70%"
), use_container_width=True)

# Export
st.subheader("Export Data")
if st.button("Download CSV", key="chw_export"):
    df = pd.DataFrame({
        "Metric": ["Referrals", "Badge Adherence", "Symptom Risk"],
        "Value": [mock_data['referrals'], mock_data['badge_adherence'], mock_data['symptom_risk']['risk']],
        "Status": ["N/A", "N/A", mock_data['symptom_risk']['status']]
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "chw_export.csv", "text/csv")