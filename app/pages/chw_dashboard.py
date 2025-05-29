import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import load_data
from utils.viz_helper import plot_annotated_line_chart, render_kpi_card, render_traffic_light, plot_treemap

# Mock data
mock_data = {
    "incidence_rate": 3.5,  # Cases per 1,000
    "referrals": 5,
    "referrals_trend": [3, 4, 5, 4, 6, 5, 5],
    "badge_adherence": 85,
    "badge_trend": [80, 82, 84, 83, 85, 84, 85],
    "risk_score": {"value": 75, "ci_lower": 70, "ci_upper": 80, "status": "Moderate", "action": "Escalate triage"},
    "risk_trend": [50, 55, 60, 65, 70, 72, 75],
    "sdoh_index": 65,  # Social determinants of health index
    "alerts": [
        {"id": "P001", "issue": "Acute BP Spike", "status": "High", "type": "BP", "priority_score": 0.9},
        {"id": "P002", "issue": "TB Risk", "status": "Moderate", "type": "TB", "priority_score": 0.6},
        {"id": "P003", "issue": "TB Risk", "status": "Moderate", "type": "TB", "priority_score": 0.5}
    ],
    "risk_zones": [{"zone": "Zone A", "issue": "High CO₂", "status": "High"}]
}

# Date labels
date_labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"]

st.header("CHW Dashboard")
st.markdown("**Actionable Field Insights for Community Health Workers**")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("Incidence Rate", f"{mock_data['incidence_rate']}/1,000", "📈", status="Moderate")
with col2:
    render_kpi_card("Pending Referrals", f"{mock_data['referrals']} patients", "📋")
with col3:
    render_kpi_card("Risk Score", f"{mock_data['risk_score']['value']}% ({mock_data['risk_score']['ci_lower']}-{mock_data['risk_score']['ci_upper']}%)", "⚠️", status=mock_data['risk_score']['status'])
with col4:
    render_kpi_card("SDOH Index", f"{mock_data['sdoh_index']}%", "🏘️", status="Moderate")

# Alerts
st.subheader("Critical Alerts")
for alert in mock_data['alerts']:
    render_traffic_light(f"{alert['id']}: {alert['issue']} (Priority: {alert['priority_score']:.2f})", alert['status'])

# Alert Prioritization
st.subheader("Alert Prioritization")
try:
    alert_df = pd.DataFrame(mock_data['alerts'])
    if not alert_df.empty:
        st.plotly_chart(plot_treemap(
            alert_df["id"],
            alert_df["priority_score"],
            alert_df["type"],
            "Alert Prioritization by Type"
        ), use_container_width=True)
    else:
        st.warning("No alert data available for treemap.")
except Exception as e:
    st.error(f"Error rendering treemap: {str(e)}")

# Risk Zones
st.subheader("High-Risk Zones")
for zone in mock_data['risk_zones']:
    render_traffic_light(f"{zone['zone']}: {zone['issue']}", zone['status'])

# Trends
st.subheader("Trends")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_annotated_line_chart(
        date_labels,
        mock_data['referrals_trend'],
        "Referrals Trend",
        "#3b82f6",
        target_line=5,
        target_label="Target: 5"
    ), use_container_width=True)
with col2:
    st.plotly_chart(plot_annotated_line_chart(
        date_labels,
        mock_data['badge_trend'],
        "Badge Adherence (%)",
        "#22c55e",
        target_line=90,
        target_label="Target: 90%"
    ), use_container_width=True)

# AI Risk Prediction
st.subheader("AI Risk Prediction")
st.markdown(f"**{mock_data['risk_score']['action']}**")
st.plotly_chart(plot_annotated_line_chart(
    date_labels,
    mock_data['risk_trend'],
    "AI Risk Score (%)",
    "#facc15",
    target_line=70,
    target_label="Alert: 70%"
), use_container_width=True)

# Action List
st.subheader("Prioritized Actions")
actions = [
    "Triage P001 for BP Spike (High Priority)",
    "Follow-up on TB Risk for P002, P003",
    "Coordinate with Zone A for CO₂ mitigation"
]
for action in actions:
    st.markdown(f"- {action}")

# Export
st.subheader("Export Report")
if st.button("Download Report", key="chw_export"):
    df = pd.DataFrame({
        "Metric": ["Incidence Rate", "Referrals", "Risk Score", "SDOH Index"],
        "Value": [mock_data['incidence_rate'], mock_data['referrals'], mock_data['risk_score']['value'], mock_data['sdoh_index']],
        "Status": ["Moderate", "N/A", mock_data['risk_score']['status'], "Moderate"]
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "chw_report.csv", "text/csv")
