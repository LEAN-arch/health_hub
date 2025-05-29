# health_hub/config/app_config.py
import os
import pandas as pd

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # health_hub directory

# Data sources directory
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

# Assets directory
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS = os.path.join(ASSETS_DIR, "style.css")
APP_LOGO = os.path.join(ASSETS_DIR, "logo.png") # Ensure this logo.png exists in assets

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "2.0.1" # Incremented version for overhaul
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative. All Rights Reserved."
CONTACT_EMAIL = "support@healthhub-demo.com"
CACHE_TTL_SECONDS = 3600 # Cache data for 1 hour

# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 1 # For single day views like CHW daily tasks
DEFAULT_DATE_RANGE_DAYS_TREND = 30 # For trend charts
RISK_THRESHOLDS = {
    "high": 75,
    "moderate": 60,
    "low": 40, # Added for more granularity
    "chw_alert_high": 80, # CHW specific high risk threshold
    "chw_alert_moderate": 65, # CHW specific moderate risk threshold
    "district_zone_high_risk": 70 # Avg risk score for a zone to be considered high risk
}
CRITICAL_SUPPLY_DAYS = 10 # Increased critical days threshold for stock
TARGET_TEST_TURNAROUND_DAYS = 2 # Days
TARGET_PATIENT_RISK_SCORE = 50 # Target to keep patient risk scores below this average

# Disease-Specific Targets/Thresholds (as percentages or rates)
TARGET_TB_CASE_DETECTION_RATE = 85 # % of estimated cases
TARGET_MALARIA_POSITIVITY_RATE = 5 # % target to be below for RDTs
TARGET_HIV_LINKAGE_TO_CARE = 90 # % of newly diagnosed linked within X days
TARGET_HPV_SCREENING_COVERAGE = 70 # % of eligible women screened
TARGET_ANEMIA_PREVALENCE_WOMEN = 15 # Target % to be below for women of reproductive age
PNEUMONIA_CASE_FATALITY_TARGET = 5 # Target % to be below
STI_SYNDROMIC_MANAGEMENT_ACCURACY = 90 # % of cases managed according to guidelines

KEY_CONDITIONS_FOR_TRENDS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'STI-Syphilis', 'STI-Gonorrhea', 'Anemia', 'Dengue', 'Hypertension', 'Diabetes', 'Wellness Visit']
CRITICAL_TESTS_PENDING = ['Sputum-AFB', 'Sputum-GeneXpert', 'HIV-Rapid', 'HIV-ViralLoad', 'RPR', 'NAAT-GC', 'PapSmear', 'Glucose Test']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Metformin', 'Amlodipine', 'Insulin']


# Wearable/Sensor-Based Thresholds/Targets
SKIN_TEMP_FEVER_THRESHOLD_C = 38.0
SPO2_LOW_THRESHOLD_PCT = 94
SPO2_CRITICAL_THRESHOLD_PCT = 90 # Added for more severe alerts
TARGET_DAILY_STEPS = 8000
TARGET_SLEEP_HOURS = 7.0
TARGET_SLEEP_SCORE_PCT = 75 # e.g. Fitbit sleep score
STRESS_LEVEL_HIGH_THRESHOLD = 7 # out of 10

# Clinic Environment IoT Thresholds/Targets
CO2_LEVEL_ALERT_PPM = 1000
CO2_LEVEL_IDEAL_PPM = 800 # Target to stay below
PM25_ALERT_UGM3 = 25 # WHO daily guideline is often around 15-25 ug/m3
PM25_IDEAL_UGM3 = 12 # Target for cleaner air
VOC_INDEX_ALERT = 200 # Example VOC index
NOISE_LEVEL_ALERT_DB = 65 # Max desirable for waiting/consultation rooms
TARGET_WAITING_ROOM_OCCUPANCY = 10 # Max people desirable (example for a small clinic area)
TARGET_PATIENT_THROUGHPUT_PER_HOUR = 8 # Patients per hour per provider/room (example)
TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER = 5 # Example metric for hygiene compliance

# Intervention thresholds for district_dashboard.py
INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD = 60 # % score below which is concerning
INTERVENTION_TB_BURDEN_HIGH_THRESHOLD = 5 # cases (absolute or per 1000, depends on context)
INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD = 10 # cases (absolute or per 1000)
INTERVENTION_PREVALENCE_HIGH_PERCENTILE = 0.75 # Top 25% for prevalence considered "high"

# Plotly specific settings
DEFAULT_PLOT_HEIGHT = 400 # Standard height for most plots
COMPACT_PLOT_HEIGHT = 320 # For smaller, denser dashboard sections
MAP_PLOT_HEIGHT = 600 # Taller height for maps

# Map Configurations
TIJUANA_CENTER_LAT = 32.5149
TIJUANA_CENTER_LON = -117.0382
TIJUANA_DEFAULT_ZOOM = 10 # Adjusted for potentially wider district view
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM
DEFAULT_CRS = "EPSG:4326" # WGS84
MAPBOX_STYLE = "carto-positron"  # Cleaner default; for other styles, MAPBOX_ACCESS_TOKEN env var is needed.
                                 # Other options: "open-street-map", "satellite-streets" (if token set)
                                 # "mapbox://styles/your_username/your_style_id" for custom Mapbox styles.

# Logging Configuration
LOG_LEVEL = "INFO" # Set to DEBUG for more verbose output during development
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S' # Added for consistency in asctime

# Color Palette (Example - for consistent categorical coloring in plots)
DISEASE_COLORS = {
    "TB": "#EF4444",        # Red
    "Malaria": "#F59E0B",   # Amber
    "HIV-Positive": "#8B5CF6", # Violet
    "Pneumonia": "#3B82F6", # Blue
    "Anemia": "#10B981",    # Green
    "STI": "#EC4899",       # Pink
    "Dengue": "#6366F1",    # Indigo
    "Hypertension": "#F97316", # Orange
    "Diabetes": "#0EA5E9",   # Sky Blue
    "Wellness Visit": "#84CC16", # Lime
    "Other": "#6B7280"      # Gray
}
RISK_STATUS_COLORS = {
    "High": "#EF4444",
    "Moderate": "#F59E0B",
    "Low": "#10B981",
    "Neutral": "#6B7280"
}

# Ensure all necessary directories exist (optional, good for robustness if app creates files)
# Path(ASSETS_DIR).mkdir(parents=True, exist_ok=True)
# Path(DATA_SOURCES_DIR).mkdir(parents=True, exist_ok=True)
