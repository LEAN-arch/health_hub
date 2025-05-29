import pytest
import pandas as pd
from utils.data_processor import load_data, load_geojson

def test_load_data():
    df = load_data("data/sample_data.csv")
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Data should not be empty"

def test_load_geojson():
    geojson = load_geojson("data/zones.geojson")
    assert geojson is not None, "GeoJSON should not be None"
    assert "features" in geojson, "GeoJSON should have features"