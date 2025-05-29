import pandas as pd
import json
import os
import geopandas as gpd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load and validate CSV data.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        # Validate required columns
        required = ["id", "zone", "condition", "risk", "population"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")
        logger.info(f"Loaded CSV data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def load_geojson(file_path):
    """
    Load and validate GeoJSON data.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        gdf = gpd.read_file(file_path)
        # Validate required properties
        required = ["zone", "risk", "facilities", "population"]
        if not all(prop in gdf.columns for prop in required):
            raise ValueError(f"Missing required GeoJSON properties: {required}")
        logger.info(f"Loaded GeoJSON data from {file_path}")
        return gdf.__geo_interface__
    except Exception as e:
        logger.error(f"Error loading GeoJSON: {str(e)}")
        return None

def calculate_incidence_rate(df, population_col="population", case_col="risk"):
    """
    Calculate incidence rate per 1,000.
    """
    try:
        total_cases = df[case_col].sum()
        total_population = df[population_col].sum()
        rate = (total_cases / total_population) * 1000 if total_population > 0 else 0
        logger.info(f"Calculated incidence rate: {rate:.2f}")
        return rate
    except Exception as e:
        logger.error(f"Error calculating incidence rate: {str(e)}")
        return 0.0
