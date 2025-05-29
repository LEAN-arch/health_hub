import pandas as pd
import streamlit as st
import json

def load_data(file_path):
    """
    Load CSV data with error handling.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.warning("Loaded data is empty.")
        return df
    except FileNotFoundError:
        st.error(f"Data file {file_path} not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def load_geojson(file_path):
    """
    Load GeoJSON data with error handling.
    """
    try:
        with open(file_path, "r") as f:
            geojson = json.load(f)
        return geojson
    except FileNotFoundError:
        st.error(f"GeoJSON file {file_path} not found.")
        return None
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        return None