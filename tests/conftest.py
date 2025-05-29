# tests/conftest.py
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon # Added Polygon
from config import app_config # To access RISK_THRESHOLDS, etc. in tests if needed

@pytest.fixture(scope="session") # scope="session" means this runs once per test session
def sample_health_records_df_main():
    """
    Provides a more comprehensive DataFrame of health records for testing various functions.
    Includes columns expected by multiple processing functions.
    """
    data = {
        'date': pd.to_datetime([
            '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03',
            '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-05', '2023-01-06'
        ]),
        'zone_id': [
            'ZN001', 'ZN002', 'ZN001', 'ZN003', 'ZN002',
            'ZN001', 'ZN003', 'ZN001', 'ZN002', 'ZN004'
        ],
        'patient_id': [
            'P001', 'P002', 'P003', 'P004', 'P005',
            'P001', 'P006', 'P007', 'P002', 'P008'
        ],
        'age': [45, 30, 62, 25, 55, 45, 38, 28, 30, 50],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male'],
        'condition': [
            'Hypertension', 'TB', 'Diabetes', 'Malaria', 'TB',
            'Hypertension', 'ARI', 'Diabetes', 'TB', 'Hypertension'
        ],
        'chw_visit': [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        'referral_status': [
            'Pending', 'Completed', 'N/A', 'Pending', 'Initiated',
            'Completed', 'N/A', 'Completed', 'Follow-up', 'Pending'
        ],
        'referral_date': pd.to_datetime([
            '2023-01-01', '2022-12-28', None, '2023-01-02', '2023-01-03',
            '2023-01-01', None, '2023-01-01', '2022-12-28', '2023-01-06'
        ]),
        'test_type': [
            'N/A', 'Sputum', 'Glucose', 'RDT', 'Sputum',
            'N/A', 'N/A', 'Glucose', 'N/A', 'N/A'
        ],
        'test_result': [
            'N/A', 'Positive', 'High', 'Pending', 'Pending',
            'N/A', 'N/A', 'Normal', 'N/A', 'N/A'
        ],
        'test_date': pd.to_datetime([
            None, '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03',
            None, None, '2023-01-03', None, None
        ]),
        'test_turnaround_days': [None, 3.0, 0.0, None, None, None, None, 0.0, None, None],
        'supply_item': [
            'BP_Med_A', 'TB_Drug_Kit', 'Insulin', 'ACT_Meds', 'TB_Drug_Kit',
            'BP_Med_A', 'Amoxicillin', 'Insulin', 'TB_Drug_Kit', 'BP_Med_B'
        ],
        'supply_level_days': [25, 15, 30, 20, 14, 22, 10, 28, 12, 18],
        'ai_risk_score': [65, 80, 70, 55, 75, 60, 40, 50, 82, 72],
        'tb_contact_traced': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_attributes_df_main():
    """Provides a DataFrame of zone attributes."""
    data = {
        'zone_id': ['ZN001', 'ZN002', 'ZN003', 'ZN004'],
        'name': ['Northwood', 'Southville', 'Eastgate', 'Westend'],
        'population': [15000, 25000, 18000, 22000],
        'socio_economic_index': [0.65, 0.40, 0.75, 0.50],
        'num_clinics': [2, 1, 3, 2],
        'avg_travel_time_clinic_min': [15, 25, 10, 20]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main():
    """Provides a GeoDataFrame of zone geometries."""
    data = {'zone_id': ['ZN001', 'ZN002', 'ZN003', 'ZN004']}
    # Simple square polygons for testing
    geometries = [
        Polygon([[0,0],[0,10],[10,10],[10,0],[0,0]]),
        Polygon([[10,0],[10,10],[20,10],[20,0],[10,0]]),
        Polygon([[0,-10],[0,0],[10,0],[10,-10],[0,-10]]),
        Polygon([[10,-10],[10,0],[20,0],[20,-10],[10,-10]])
    ]
    return gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")

@pytest.fixture
def empty_health_df():
    return pd.DataFrame(columns=[ # Ensure it has columns expected by some functions to avoid KeyErrors
        'date', 'zone_id', 'patient_id', 'condition', 'ai_risk_score',
        'referral_status', 'test_result', 'test_date', 'test_turnaround_days',
        'supply_level_days', 'tb_contact_traced', 'chw_visit'
    ])


@pytest.fixture
def empty_gdf():
    return gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs="EPSG:4326")
