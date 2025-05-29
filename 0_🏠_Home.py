import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="Health Intelligence Hub - Home",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load CSS
def load_css(file_name):
    abs_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(abs_path):
        with open(abs_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            # print(f"Loaded CSS from {abs_path}") # For debugging
    else:
        st.warning(f"CSS file not found: {abs_path}")

# Load custom CSS
load_css("style.css")


st.title("WELCOME TO THE COMMUNITY HEALTH INTELLIGENCE HUB")
st.markdown("---")

st.markdown(
    """
    **Empowering health stakeholders with actionable, real-time insights for improved community well-being.**

    This platform offers a suite of tools designed to enhance decision-making at various levels of the public health system. 
    By leveraging data analytics and visualization, the Health Hub aims to provide timely and relevant information to:
    """
)

cols = st.columns(3)
with cols[0]:
    st.subheader("🧑‍⚕️ Community Health Workers")
    st.markdown("- Prioritize patient follow-ups with AI-driven risk scores.\n- Track field activities and referrals effectively.\n- Access critical patient alerts on the go.")
with cols[1]:
    st.subheader("🏥 Clinic Managers")
    st.markdown("- Monitor operational efficiency and patient flow.\n- Manage test turnaround times and supply chains.\n- Identify service quality gaps and resource needs.")
with cols[2]:
    st.subheader("🗺️ District Officers")
    st.markdown("- Analyze population health trends and disease hotspots.\n- Assess intervention impacts and facility coverage.\n- Plan resource allocation strategically.")

st.markdown("---")
st.info("👈 **Please select your role-specific dashboard from the sidebar to explore tailored insights and tools.**")

st.sidebar.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/components/extras/multipage_apps/mpa_example_files/logo.png", width=100) # Placeholder logo
st.sidebar.markdown("## Health Hub Navigation")
st.sidebar.markdown("Use the links above to access dashboards.")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for enhanced public health decision-making.")
st.sidebar.markdown("Version 1.0.0")
# Add source code link if public
# st.sidebar.markdown("[View Source Code](YOUR_GITHUB_REPO_LINK)")

# Example of how to use an expander for more info
with st.expander("About the Data & Methodology", expanded=False):
    st.markdown("""
        - **Data Sources:** Anonymized data is sourced from routine health information systems, community health worker reports, and geospatial databases. 
          For this demonstration, synthetic data is used.
        - **Risk Scoring:** AI-driven risk scores are generated using predictive models that consider various health and socio-demographic factors (simulated in this demo).
        - **Privacy:** All data displayed is aggregated or anonymized to protect patient privacy in a real-world scenario.
        - **Updates:** In a production system, data would be updated regularly (e.g., daily or weekly) to ensure timeliness.
    """)