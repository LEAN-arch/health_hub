# config/app_config.py
import os
import pandas as pd

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # health_hub_app directory

# Data sources directory
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

# Assets directory
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS = os.path.join(ASSETS_DIR, "style.css")
APP_LOGO = os.path.join(ASSETS_DIR, "logo.png") # Ensure this logo.png exists

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "1.3.1" # Incremented version
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative"
CONTACT_EMAIL = "support@healthhub-demo.com"

# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 1
DEFAULT_DATE_RANGE_DAYS_TREND = 30
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

# Disease-Specific Targets/Thresholds
TARGET_TB_CASE_DETECTION_RATE = 85
TARGET_MALARIA_POSITIVITY_RATE = 5
TARGET_HIV_LINKAGE_TO_CARE = 90
TARGET_HPV_SCREENING_COVERAGE = 70
TARGET_ANEMIA_PREVALENCE_WOMEN = 15 # Target % to be below
PNEUMONIA_CASE_FATALITY_TARGET = 5 # Target % to be below
STI_SYNDROMIC_MANAGEMENT_ACCURACY = 90

KEY_CONDITIONS_FOR_TRENDS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'STI-Syphilis', 'STI-Gonorrhea', 'Anemia', 'Dengue']
CRITICAL_TESTS_PENDING = ['Sputum-AFB', 'Sputum-GeneXpert', 'HIV-Rapid', 'RPR', 'NAAT-GC', 'PapSmear'] # Added PapSmear

# New Sensor-Based Thresholds/Targets
SKIN_TEMP_FEVER_THRESHOLD_C = 38.0
SPO2_LOW_THRESHOLD_PCT = 94
TARGET_DAILY_STEPS = 7000
TARGET_SLEEP_HOURS = 7.0
TARGET_SLEEP_SCORE_PCT = 75
STRESS_LEVEL_HIGH_THRESHOLD = 7 # out of 10

CO2_LEVEL_ALERT_PPM = 1000
CO2_LEVEL_IDEAL_PPM = 800
PM25_ALERT_UGM3 = 25
PM25_IDEAL_UGM3 = 12
VOC_INDEX_ALERT = 200
NOISE_LEVEL_ALERT_DB = 65
TARGET_WAITING_ROOM_OCCUPANCY = 10 # Max people desirable (example)
TARGET_PATIENT_THROUGHPUT_PER_HOUR = 8 # Patients per hour per doctor/room (example)
TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER = 5 # Example

# Intervention thresholds (as discussed for district_dashboard.py)
INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD = 50
INTERVENTION_TB_BURDEN_HIGH_THRESHOLD = 5
INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD = 10


# Plotly specific settings
DEFAULT_PLOT_HEIGHT = 380
MAP_PLOT_HEIGHT = 575
TIJUANA_CENTER_LAT = 32.5149
TIJUANA_CENTER_LON = -117.0382
TIJUANA_DEFAULT_ZOOM = 11
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM
DEFAULT_CRS = "EPSG:4326" # Added for consistency if used by GDF creation

# --- Map Configurations --- <<< Section already exists, just add MAPBOX_STYLE here
MAPBOX_STYLE = "open-street-map"  # Or "carto-positron", "carto-darkmatter", etc.
                                  # Or your custom Mapbox style URL: "mapbox://styles/your_username/your_style_id"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
