# app_home.py
import streamlit as st
import os
import pandas as pd # Moved import here as it's used for APP_FOOTER
from config import app_config # Import the new config

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"{app_config.APP_TITLE} - Home",
    page_icon=app_config.APP_LOGO if os.path.exists(app_config.APP_LOGO) else "❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@example-health-hub.com', # Replace with actual link
        'Report a bug': "mailto:bugs@example-health-hub.com", # Replace
        'About': f"""
        **{app_config.APP_TITLE}** (v{app_config.APP_VERSION})
        {app_config.APP_FOOTER}
        This application provides data-driven insights for public health professionals.
        For demonstration purposes only using synthetic data.
        """
    }
)

# --- Function to load CSS ---
def load_css(css_file_path):
    if os.path.exists(css_file_path):
        with open(css_file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Successfully loaded CSS from {css_file_path}")
    else:
        # This warning might be annoying if CSS is optional or path is wrong in dev
        # For production, ensure path is correct.
        logger.warning(f"CSS file not found at {css_file_path}. Styling may be affected.")
        # st.warning(f"CSS file not found: {os.path.basename(css_file_path)}. Default styles will be used.")

# --- Logging Setup (Basic, can be more sophisticated) ---
import logging # Ensure logging is imported if not already via app_config
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO), 
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__) # Get a logger for this specific file

# Load custom CSS
load_css(app_config.STYLE_CSS)


# --- App Header ---
header_cols = st.columns([0.1, 0.9]) # Adjust column ratio for logo vs title
with header_cols[0]:
    if os.path.exists(app_config.APP_LOGO):
        st.image(app_config.APP_LOGO, width=80) # Slightly smaller logo
    # else:
    #     st.markdown("❤️", unsafe_allow_html=True) # Fallback if no logo, simple emoji

with header_cols[1]:
    st.title(app_config.APP_TITLE)
    st.caption(f"Version {app_config.APP_VERSION} | Empowering Health Decisions with Data")

st.markdown("---")

# --- App Introduction ---
st.markdown(
    """
    #### Welcome!
    This platform is dedicated to providing actionable, real-time intelligence to enhance community well-being and public health outcomes.
    Our suite of tools leverages data analytics and interactive visualizations to support decision-making at all levels of the health system.
    """
)

st.subheader("Role-Specific Dashboards:")
intro_cols = st.columns(3)
with intro_cols[0]:
    st.markdown("##### 🧑‍⚕️ Community Health Workers")
    st.markdown("<small>Focus on field insights, patient tracking, alert prioritization, and efficient task management.</small>", unsafe_allow_html=True)
with intro_cols[1]:
    st.markdown("##### 🏥 Clinic Managers")
    st.markdown("<small>Monitor operational efficiency, test turnaround times, supply chain, and service quality metrics.</small>", unsafe_allow_html=True)
with intro_cols[2]:
    st.markdown("##### 🗺️ District Officers")
    st.markdown("<small>Analyze population health trends, disease hotspots, intervention impacts, and strategic resource allocation.</small>", unsafe_allow_html=True)

st.markdown("---")
st.success("👈 **Please select a dashboard from the sidebar navigation to explore tailored insights and tools.**")

# --- Additional Home Page Content (Optional) ---
with st.expander("ℹ️ About This Platform", expanded=False):
    st.markdown(f"""
        - **Purpose:** To demonstrate a comprehensive health intelligence system.
        - **Data:** Utilizes synthetic data for illustrative purposes. In a real-world scenario, this would connect to live, anonymized Health Information Systems.
        - **Technology:** Built with Python, Streamlit, Pandas, GeoPandas, and Plotly.
        - **Key Features:** Role-based access, dynamic filtering, interactive visualizations, and downloadable reports (conceptual).
        - **Disclaimer:** This application is a prototype and should not be used for actual medical decision-making.
    """)

with st.expander("✨ Key Capabilities", expanded=True):
    cap_cols = st.columns(3)
    with cap_cols[0]:
        st.markdown("📈 **Trend Analysis**\n<small>Visualize health indicators over time.</small>", unsafe_allow_html=True)
    with cap_cols[1]:
        st.markdown("🗺️ **Geospatial Insights**\n<small>Map disease hotspots and resource distribution.</small>", unsafe_allow_html=True)
    with cap_cols[2]:
        st.markdown("🔔 **Alert Systems**\n<small>Identify high-risk patients and critical situations.</small>", unsafe_allow_html=True)
    # Add more capabilities if relevant


# --- Sidebar Content ---
# Streamlit automatically creates navigation from files in 'pages/'
# Adding custom branding or links to the sidebar:
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER)
if 'CONTACT_EMAIL' in app_config.__dict__: # Check if a contact email is in config
     st.sidebar.caption(f"Contact: [{app_config.CONTACT_EMAIL}](mailto:{app_config.CONTACT_EMAIL})")

logger.info("Home page loaded successfully.")
