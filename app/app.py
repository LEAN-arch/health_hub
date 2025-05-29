import streamlit as st

st.set_page_config(
    page_title="Health Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Community Health Intelligence Hub")
st.markdown("**Role-based insights for health stakeholders**")

# Sidebar for role selection
st.sidebar.header("Select Role")
role = st.sidebar.selectbox("Role", ["CHW", "Clinic", "District"])

# Role-based navigation
if role == "CHW":
    st.markdown("### Community Health Worker Dashboard")
    st.write("Navigate to the CHW Dashboard page for field insights.")
elif role == "Clinic":
    st.markdown("### Clinic Manager Dashboard")
    st.write("Navigate to the Clinic Dashboard page for operational insights.")
elif role == "District":
    st.markdown("### District Officer Dashboard")
    st.write("Navigate to the District Dashboard page for strategic insights.")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("Use the sidebar to select a role and navigate to the corresponding dashboard.")
st.sidebar.markdown("[Source Code](https://github.com/LEAN-arch/health_hub)")
