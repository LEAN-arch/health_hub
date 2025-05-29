# tests/conftest.py
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from config import app_config # To access RISK_THRESHOLDS, DEFAULT_CRS, etc.

@pytest.fixture(scope="session")
def sample_health_records_df_main():
    """
    Provides a comprehensive DataFrame of health records, mimicking 'health_records.csv'
    and including columns relevant to various dashboard functionalities.
    """
    data = {
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P001', 'P002', 'P009', 'P010'],
        'date': pd.to_datetime([
            '2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02', '2023-10-03',
            '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-07', '2023-10-08',
            '2023-10-09', '2023-10-10' # Added more varied dates
        ]),
        'condition': ['TB', 'Malaria', 'Wellness Visit', 'Pneumonia', 'TB', 'STI-Syphilis', 'Anemia', 'HIV-Positive', 'TB', 'Malaria', 'Hypertension', 'Diabetes'],
        'test_type': ['Sputum-AFB', 'RDT-Malaria', 'HIV-Rapid', 'Chest X-Ray', 'Sputum-GeneXpert', 'RPR', 'Hemoglobin Test', 'HIV-ViralLoad', 'Follow-up', 'RDT-Malaria', 'BP Check', 'Glucose Test'],
        'test_result': ['Positive', 'Positive', 'Negative', 'Positive', 'Positive', 'Positive', 'Low', '2500', 'N/A', 'Negative', 'High', 'High'],
        'test_turnaround_days': [3, 0, 1, 2, 2, 1, 0, 7, 0, 0, 0, 1],
        'test_date': pd.to_datetime([
            '2023-09-28', '2023-10-01', '2023-10-01', '2023-09-30', '2023-10-01',
            '2023-10-02', '2023-10-04', '2023-09-28', '2023-10-07', '2023-10-08',
            '2023-10-09', '2023-10-09'
        ]),
        'item': ['TB-Regimen A', 'ACT Tablets', 'ARV-Regimen B', 'Amoxicillin Syrup', 'TB-Regimen A', 'Penicillin G', 'Iron-Folate Tabs', 'ARV-Regimen B', 'TB-Regimen A', 'ACT Tablets', 'Amlodipine', 'Metformin'],
        'quantity_dispensed': [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
        'stock_on_hand': [100, 50, 75, 200, 98, 30, 150, 73, 96, 48, 120, 90],
        'consumption_rate_per_day': [2, 1, 0.5, 5, 2, 0.2, 3, 0.5, 2, 1, 1.5, 3],
        'zone_id': ['ZoneA', 'ZoneB', 'ZoneA', 'ZoneC', 'ZoneA', 'ZoneB', 'ZoneC', 'ZoneA', 'ZoneA', 'ZoneB', 'ZoneD', 'ZoneB'],
        'ai_risk_score': [85, 70, 30, 65, 88, 75, 55, 92, 80, 40, 72, 60],
        'avg_daily_steps': [3500, 6200, 8100, 4500, 3400, 5800, 7500, 4200, 3700, 7800, 5500, 6300],
        'avg_spo2': [97, 98, 99, 95, 96, 97, 98, 96, 97, 98, 98, 97],
        'min_spo2_pct': [95, 96, 98, 93, 94, 95, 97, 94, 95, 97, 96, 95],
        'max_skin_temp_celsius': [37.5, 37.2, 36.8, 38.5, 38.1, 37.0, 36.9, 38.2, 37.3, 37.1, 37.2, 37.3],
        'fall_detected_today': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'clinic_id': ['C01', 'C02', 'C01', 'C03', 'C01', 'C02', 'C03', 'C01', 'C01', 'C02', 'C04', 'C02'],
        'physician_id': ['DOC001', 'DOC002', 'DOC001', 'DOC003', 'DOC001', 'DOC002', 'DOC003', 'DOC001', 'DOC001', 'DOC002', 'DOC004', 'DOC002'],
        'notes': ['Follow up', '', '', '', 'Resistant strain?', '', '', 'Adherence counseling', '', '', 'New diagnosis HTN', 'Dietary advice given'],
        'hpv_status': ['Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Unknown', 'Unknown'],
        'hiv_viral_load': [None, None, None, None, None, None, None, 2500.0, None, None, None, None],
        'chw_visit': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        'referral_status': ['Pending', 'Completed', 'N/A', 'Pending', 'Initiated', 'Completed', 'N/A', 'Pending', 'Follow-up', 'Completed', 'Pending', 'Completed'],
        'referral_date': pd.to_datetime([None, '2023-09-25', None, '2023-10-01', '2023-10-02', '2023-09-28', None, '2023-10-04', '2023-10-06', '2023-10-07', None, '2023-10-09']),
        'tb_contact_traced': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male'],
        'age': [45, 30, 62, 25, 55, 38, 28, 50, 45, 30, 58, 67],
        'resting_heart_rate': [65,70,60,75,68,72,66,78,64,62,80,70],
        'avg_hrv': [50,45,55,40,48,42,52,38,51,53,35,44],
        'avg_sleep_duration_hrs': [6.5,7.2,8.0,5.0,6.0,7.5,6.8,5.5,6.7,7.8,5.8,7.1],
        'sleep_score_pct': [70,78,85,60,65,80,72,58,73,82,55,77],
        'stress_level_score': [5,3,2,6,7,4,3,8,4,2,7,5]
    }
    df = pd.DataFrame(data)
    # Ensure correct dtypes after creation for numeric columns (as they might be object if NaNs are present initially)
    numeric_cols = [
        'test_turnaround_days', 'quantity_dispensed', 'stock_on_hand', 'consumption_rate_per_day',
        'ai_risk_score', 'avg_daily_steps', 'avg_spo2', 'min_spo2_pct', 'max_skin_temp_celsius',
        'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced', 'resting_heart_rate',
        'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score', 'hiv_viral_load'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@pytest.fixture(scope="session")
def sample_iot_clinic_df_main():
    """Provides a sample DataFrame for IoT clinic environment data."""
    data = {
        'timestamp': pd.to_datetime([
            '2023-10-01T09:00:00Z', '2023-10-01T10:00:00Z', '2023-10-01T09:00:00Z', '2023-10-01T09:30:00Z',
            '2023-10-02T09:00:00Z', '2023-10-02T14:00:00Z', '2023-10-03T10:00:00Z'
        ]),
        'clinic_id': ['C01', 'C01', 'C02', 'C03', 'C01', 'C04', 'C01'],
        'room_name': ['Waiting Room A', 'Waiting Room A', 'Consultation 1', 'TB Clinic Waiting', 'Consultation 1', 'General Waiting Area', 'Waiting Room A'],
        'avg_co2_ppm': [650, 700, 550, 950, 680, 700, 660],
        'max_co2_ppm': [700, 750, 600, 1100, 720, 780, 710],
        'avg_pm25': [10.5, 12.1, 8.2, 22.8, 11.0, 13.5, 10.8],
        'voc_index': [120,130,100,180, 125, 135, 122],
        'avg_temp_celsius': [24.5, 24.7, 23.0, 24.0, 24.3, 24.6, 24.4],
        'avg_humidity_rh': [55,56,50,58, 54, 56, 55],
        'avg_noise_db': [50,52,45,53, 51, 50, 50],
        'waiting_room_occupancy': [5, 7, 2, 12, 6, 4, 5],
        'patient_throughput_per_hour': [8,9,10,5, 8, 6, 9],
        'sanitizer_dispenses_per_hour': [2,4,1,2, 3, 4, 3],
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneC', 'ZoneA', 'ZoneD', 'ZoneA'] # Explicitly adding zone_id
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_attributes_df_main():
    """Provides a DataFrame of zone attributes, matching expanded GeoJSON."""
    data = {
        'zone_id': ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE', 'ZoneF'],
        'name': ['Northwood District', 'Southville Area', 'Eastgate Community', 'Westend Borough', 'Central City Sector', 'Riverdale Precinct'],
        'population': [12500, 21000, 17500, 8500, 35000, 15000],
        'socio_economic_index': [0.65, 0.42, 0.78, 0.55, 0.85, 0.30],
        'num_clinics': [2, 1, 3, 1, 4, 1],
        'avg_travel_time_clinic_min': [15, 28, 12, 22, 8, 35]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main():
    """Provides a GeoDataFrame of zone geometries, matching expanded attributes."""
    # Ensure this matches the `zone_geometries.geojson` content
    features = [
        {"zone_id": "ZoneA", "name": "Northwood District", "polygon": Polygon([[0,0],[0,10],[10,10],[10,0],[0,0]])},
        {"zone_id": "ZoneB", "name": "Southville Area", "polygon": Polygon([[10,0],[10,10],[20,10],[20,0],[10,0]])},
        {"zone_id": "ZoneC", "name": "Eastgate Community", "polygon": Polygon([[0,-10],[0,0],[10,0],[10,-10],[0,-10]])},
        {"zone_id": "ZoneD", "name": "Westend Borough", "polygon": Polygon([[10,-10],[10,0],[20,0],[20,-10],[10,-10]])},
        {"zone_id": "ZoneE", "name": "Central City Sector", "polygon": Polygon([[-10,0],[-10,10],[0,10],[0,0],[-10,0]])}, # Added ZoneE
        {"zone_id": "ZoneF", "name": "Riverdale Precinct", "polygon": Polygon([[-10,-10],[-10,0],[0,0],[0,-10],[-10,-10]])} # Added ZoneF
    ]
    df = pd.DataFrame(features)
    return gpd.GeoDataFrame(df, geometry='polygon', crs=app_config.DEFAULT_CRS)

@pytest.fixture(scope="session")
def sample_enriched_gdf_main(sample_zone_geometries_gdf_main, sample_zone_attributes_df_main, sample_health_records_df_main, sample_iot_clinic_df_main):
    """
    Provides a sample enriched GeoDataFrame as if produced by
    `enrich_zone_geodata_with_health_aggregates`. This is a simplified mock of enrichment.
    For full testing, one would call the actual enrichment function.
    """
    from utils.core_data_processing import enrich_zone_geodata_with_health_aggregates # Import locally for fixture
    
    # Base GDF is merge of geometries and attributes
    # In a real test of enrich_zone_geodata, you'd call load_zone_data()
    base_gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left", suffixes=('', '_attr'))
    if 'name_attr' in base_gdf.columns: base_gdf['name'] = base_gdf['name_attr'] # Prioritize name from attributes
    
    # Call the actual enrichment function to get a realistic enriched GDF for other tests
    enriched_gdf = enrich_zone_geodata_with_health_aggregates(
        base_gdf,
        sample_health_records_df_main,
        sample_iot_clinic_df_main
    )
    return enriched_gdf


# Fixtures for testing plotting functions (more varied data)
@pytest.fixture(scope="session")
def sample_series_data():
    """Time series data for line charts."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'])
    values = [10, 12, 9, 15, 13, 18, 16]
    return pd.Series(values, index=dates, name="Daily Count")

@pytest.fixture(scope="session")
def sample_bar_df():
    """DataFrame for bar charts, including a grouping column."""
    data = {
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
        'value': [20, 15, 25, 30, 18, 22, 28, 12],
        'group': ['G1', 'G1', 'G2', 'G1', 'G2', 'G2', 'G1', 'G1']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_donut_df():
    """DataFrame for donut charts."""
    data = {'status': ['Critical', 'Warning', 'Okay', 'Unknown'], 'count': [5, 12, 30, 3]}
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_heatmap_df():
    """DataFrame suitable for heatmaps (e.g., correlation matrix style)."""
    np.random.seed(42) # for reproducibility
    data = np.random.rand(5, 5) * 2 - 1 # Values between -1 and 1
    df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(5)], index=[f'Factor{j+1}' for j in range(5)])
    df.iloc[0,0] = 1.0; df.iloc[1,1] = 1.0; df.iloc[2,2]=1.0; df.iloc[3,3]=1.0; df.iloc[4,4]=1.0 # Diagonals
    return df

@pytest.fixture(scope="session")
def sample_choropleth_gdf(sample_zone_geometries_gdf_main, sample_zone_attributes_df_main):
    """GeoDataFrame for choropleth maps, merged with some basic attributes."""
    gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    # Add a sample numeric column for coloring the map
    np.random.seed(10)
    gdf['risk_score'] = np.random.randint(30, 90, size=len(gdf))
    if 'name_y' in gdf.columns: gdf.rename(columns={'name_y':'name'}, inplace=True) # Fix potential duplicate name col
    return gdf


# Fixtures for empty DataFrames/GeoDataFrames with expected schemas for robust error testing
@pytest.fixture
def empty_health_df_with_schema():
    return pd.DataFrame(columns=[
        'patient_id', 'date', 'condition', 'test_type', 'test_result', 'test_turnaround_days',
        'item', 'quantity_dispensed', 'stock_on_hand', 'consumption_rate_per_day', 'zone_id',
        'ai_risk_score', 'avg_daily_steps', 'avg_spo2', 'min_spo2_pct', 'max_skin_temp_celsius',
        'fall_detected_today', 'clinic_id', 'referral_status', 'gender', 'age'
    ])

@pytest.fixture
def empty_iot_df_with_schema():
    return pd.DataFrame(columns=[
        'timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25',
        'avg_temp_celsius', 'waiting_room_occupancy', 'zone_id'
    ])

@pytest.fixture
def empty_zone_attributes_df_with_schema():
     return pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min'])


@pytest.fixture
def empty_gdf_with_schema():
    return gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs=app_config.DEFAULT_CRS)
