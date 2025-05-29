import streamlit as st

# Set page configuration
st.set_page_config(page_title="Health Intelligence Hub", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("Community Health Intelligence Hub")

# Sidebar for role selection
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("Select a role to view tailored insights.")
st.sidebar.markdown("---")
role = st.sidebar.selectbox("Select Role", ["CHW", "Clinic", "District"], key="role_selector")

# Instructions
st.markdown(f"Welcome to the **{role} Dashboard**. Navigate using the sidebar or explore pages below.")