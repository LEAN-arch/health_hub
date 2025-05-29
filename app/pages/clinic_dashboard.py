import streamlit as st
import pandas as pd
from utils.data_processor import load_data
from utils.viz_helper import plot_line_chart, render_kpi_card, render_traffic_light, plot_donut_chart

# Mock data
mock_data = {
    "flagged_total": 15,
    "flagged_trend": [10, 12, 15, 14, 15, 15, 15],
    "positivity_rate": 12,
    "test_rate": {"tests": 50, "positive": 10, "time_to_result": 2.3, "negative": 35, "pending": 5},
    "result_trend": [2.5, 2.4, 2.3, 2.3, 2.2, 2.3, 2.3],
    "supply": {"tests": 120, "stock_days": 2, "status": "High", "forecast": [120, 100, 80, 50, 20, 0]},
    "uptime": 90,
    "chw_performance": 1.7,
    "chw_trend": [2.0, 1.9, 1.8, 1.7, 1.7, 1.6, 1.7]
}

# Date labels
date_labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"]

st.header("Clinic Dashboard")
st.markdown("**Operational insights for Clinic Managers**")

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    render_kpi_card("Flagged Individuals", f"{mock_data['flagged_total']}", "🚨", drilldown=True)
with col2:
    render_kpi_card("Positivity Rate", f"{mock_data['positivity_rate']}%", "📊", status="Moderate" if mock_data['positivity_rate'] > 10 else "Low")
with col3:
    render_kpi_card("Device Uptime", f"{mock_data['uptime']}%", "🖥️", status="Moderate" if mock_data['uptime'] < 90 else "Low")

# Test Results
st.subheader("Test Results")
st.markdown(f"**Tests**: {mock_data['test_rate']['tests']}, **Positive**: {mock_data['test_rate']['positive']}, **Avg Time**: {mock_data['test_rate']['time_to_result']} days")
st.plotly_chart(plot_line_chart(
    date_labels,
    mock_data['result_trend'],
    "Result Time (days)",
    "#f97316",
    target_line=2.5,
    target_label="Target: 2.5 days"
), use_container_width=True)

# Test Distribution
st.subheader("Test Result Distribution")
test_dist = {
    "Positive": mock_data['test_rate']['positive'],
    "Negative": mock_data['test_rate']['negative'],
    "Pending": mock_data['test_rate']['pending']
}
st.plotly_chart(plot_donut_chart(
    list(test_dist.keys()),
    list(test_dist.values()),
    "Test Results"
), use_container_width=True)

# Supply Forecast
st.subheader("AI Supply Forecast")
st.markdown(f"**Stock**: {mock_data['supply']['tests']} tests, **Stock-Out**: {mock_data['supply']['stock_days']} days")
render_traffic_light("Supply Status", mock_data['supply']['status'])
st.plotly_chart(plot_line_chart(
    ["Today", "D+1", "D+2", "D+3", "D+4", "D+5"],
    mock_data['supply']['forecast'],
    "Supply Forecast (Tests)",
    "#ef4444",
    target_line=50,
    target_label="Critical: 50"
), use_container_width=True)

# CHW Performance
st.subheader("CHW Performance")
st.plotly_chart(plot_line_chart(
    date_labels,
    mock_data['chw_trend'],
    "Referral Time (days)",
    "#22c55e",
    target_line=1.8,
    target_label="Target: 1.8 days"
), use_container_width=True)

# Export
st.subheader("Export Data")
if st.button("Download CSV", key="clinic_export"):
    df = pd.DataFrame({
        "Metric": ["Flagged Total", "Positivity Rate", "Uptime"],
        "Value": [mock_data['flagged_total'], mock_data['positivity_rate'], mock_data['uptime']],
        "Status": [
            "High" if mock_data['flagged_total'] > 20 else "Moderate",
            "Moderate" if mock_data['positivity_rate'] > 10 else "Low",
            "Moderate" if mock_data['uptime'] < 90 else "Low"
        ]
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "clinic_export.csv", "text/csv")

# Drill-down
if st.button("Drill-Down Flagged Individuals", key="clinic_drilldown"):
    st.markdown("**Breakdown**: TB: 5, Malaria: 7, Anemia: 3")