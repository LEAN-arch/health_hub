import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import load_data
from utils.viz_helper import plot_annotated_line_chart, render_kpi_card, plot_donut_chart

# Mock data
mock_data = {
    "test_results": [
        {"status": "Positive", "count": 12},
        {"status": "Negative", "count": 25},
        {"status": "Pending", "count": 8}
    ],
    "test_turnaround": 2.5,  # Days
    "case_fatality_ratio": 0.02,  # 2%
    "supply_forecast": [50, 45, 40, 38, 35, 32, 30],
    "supply_ci_lower": [48, 43, 38, 36, 33, 30, 28],
    "supply_ci_upper": [52, 47, 42, 40, 37, 34, 32],
    "supply_risk_score": 60,  # ML-based risk
    "flagged_individuals": [
        {"id": "P004", "condition": "TB Positive", "status": "High", "action": "Immediate Follow-up"},
        {"id": "P005", "condition": "BP Anomaly", "status": "Moderate", "action": "Monitor"}
    ]
}

# Date labels
date_labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"]

st.header("Clinic Dashboard")
st.markdown("**Operational Insights for Clinic Managers**")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("Pending Tests", f"{mock_data['test_results'][2]['count']} tests", "🧪")
with col2:
    render_kpi_card("Turnaround Time", f"{mock_data['test_turnaround']} days", "⏱️", status="Moderate")
with col3:
    render_kpi_card("Case Fatality Ratio", f"{mock_data['case_fatality_ratio']*100:.1f}%", "📉", status="Low")
with col4:
    render_kpi_card("Supply Risk Score", f"{mock_data['supply_risk_score']}%", "📦", status="Moderate")

# Test Results
st.subheader("Test Results")
try:
    test_df = pd.DataFrame(mock_data['test_results'])
    if not test_df.empty:
        st.plotly_chart(plot_donut_chart(
            test_df["status"],
            test_df["count"],
            "Test Result Distribution"
        ), use_container_width=True)
    else:
        st.warning("No test result data available.")
except Exception as e:
    st.error(f"Error rendering donut chart: {str(e)}")

# Supply Forecast
st.subheader("Supply Forecast")
st.plotly_chart(plot_annotated_line_chart(
    date_labels,
    mock_data['supply_forecast'],
    "Remaining Supply (days)",
    "#22c55e",
    target_line=30,
    target_label="Critical: 30 days",
    ci_lower=mock_data['supply_ci_lower'],
    ci_upper=mock_data['supply_ci_upper']
), use_container_width=True)

# Flagged Individuals
st.subheader("Flagged Individuals")
if st.button("Drill-Down Flagged Individuals", key="clinic_drilldown"):
    flagged_df = pd.DataFrame(mock_data['flagged_individuals'])
    st.dataframe(flagged_df, use_container_width=True)

# Supply Actions
st.subheader("Supply Actions")
actions = [
    f"Reorder {int(mock_data['supply_forecast'][-1]*0.2)} units for Zone A",
    "Redistribute excess from Zone B to Zone C"
]
for action in actions:
    st.markdown(f"- {action}")

# Export
st.subheader("Export Report")
if st.button("Download Report", key="clinic_export"):
    df = pd.DataFrame({
        "Metric": ["Pending Tests", "Turnaround Time", "Case Fatality Ratio", "Supply Risk"],
        "Value": [mock_data['test_results'][2]['count'], mock_data['test_turnaround'], mock_data['case_fatality_ratio']*100, mock_data['supply_risk_score']],
        "Status": ["N/A", "Moderate", "Low", "Moderate"]
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "clinic_report.csv", "text/csv")
