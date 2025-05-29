# config/app_config.py
import os
import pandas as pd

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
APP_VERSION = "1.2.0"
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative"
CONTACT_EMAIL = "support@healthhub-demo.com"

# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 7
DEFAULT_DATE_RANGE_DAYS_TREND = 90
RISK_THRESHOLDS = {
    "high": 75,
    "moderate": 60,
    "chw_alert_high": 80,
    "chw_alert_moderate": 65,
    "district_zone_high_risk": 70
}
CRITICAL_SUPPLY_DAYS = 7
TARGET_TEST_TURNAROUND_DAYS = 2
TARGET_PATIENT_RISK_SCORE = 60

# Plotly specific settings
DEFAULT_PLOT_HEIGHT = 380
MAP_PLOT_HEIGHT = 575

# --- Tijuana Focus ---
TIJUANA_CENTER_LAT = 32.5149  # <<<<<<<<<<<< ENSURE THIS LINE EXISTS AND IS SPELLED CORRECTLY
TIJUANA_CENTER_LON = -117.0382 # <<<<<<<<<<<< ENSURE THIS LINE EXISTS AND IS SPELLED CORRECTLY
TIJUANA_DEFAULT_ZOOM = 11    # <<<<<<<<<<<< ENSURE THIS LINE EXISTS AND IS SPELLED CORRECTLY

# Use these for map defaults if no other center/zoom is specified BY THE CALLING FUNCTION
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
