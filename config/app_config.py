# config/app_config.py
import os

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # health_hub_app directory

# Data sources directory
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")

# Assets directory
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS = os.path.join(ASSETS_DIR, "style.css")
APP_LOGO = os.path.join(ASSETS_DIR, "logo.png")

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "1.1.0"
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative"


# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 7 # For point-in-time views like CHW daily tasks
DEFAULT_DATE_RANGE_DAYS_TREND = 90 # For trend analysis
RISK_THRESHOLDS = {
    "high": 75,
    "moderate": 60,
    "chw_alert_high": 80, # Specific for CHW alert highlighting
    "chw_alert_moderate": 65,
    "district_zone_high_risk": 70 # Avg risk score for a zone to be considered high risk
}
CRITICAL_SUPPLY_DAYS = 7
TARGET_TEST_TURNAROUND_DAYS = 2
TARGET_PATIENT_RISK_SCORE = 60 # Example overall target for district

# Plotly specific settings (can be expanded)
DEFAULT_PLOT_HEIGHT = 380
MAP_PLOT_HEIGHT = 550
MAP_DEFAULT_ZOOM = 7 # Adjust based on your GeoJSON extent

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Import pandas here if used for APP_FOOTER for cleaner access elsewhere
import pandas as pd
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative"
