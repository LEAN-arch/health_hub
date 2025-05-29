**Example adaptation for `test_load_zone_data`:**
```python
# In tests/test_core_data_processing.py
# ... other imports and fixtures from previous test_data_processor.py ...
# Make sure sample_zone_attributes_df and sample_zone_geometries_gdf are defined in conftest.py

@patch('utils.core_data_processing.os.path.exists')
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.gpd.read_file')
def test_load_zone_data_valid(mock_gpd_read, mock_pd_read, mock_os_exists, 
                              sample_zone_attributes_df, sample_zone_geometries_gdf, mocker):
    # Mock os.path.exists to return True for both files
    mock_os_exists.side_effect = [True, True] # First call for attributes, second for geometries
    
    mock_pd_read.return_value = sample_zone_attributes_df.copy()
    mock_gpd_read.return_value = sample_zone_geometries_gdf.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')

    merged_gdf = load_zone_data()

    assert merged_gdf is not None
    assert not merged_gdf.empty
    assert 'name' in merged_gdf.columns # From attributes
    assert 'geometry' in merged_gdf.columns # From geometries
    assert 'population' in merged_gdf.columns
    
    # Check if merge happened correctly based on zone_id
    assert len(merged_gdf) == len(sample_zone_attributes_df) # Assuming all zone_ids match
    assert merged_gdf[merged_gdf['zone_id'] == 'ZN001'].iloc[0]['name'] == 'North'
    
    mock_pd_read.assert_called_once_with(app_config.ZONE_ATTRIBUTES_CSV)
    mock_gpd_read.assert_called_once_with(app_config.ZONE_GEOMETRIES_GEOJSON)
    mock_st_error.assert_not_called()

# Add tests for cases where one file is missing, or zone_id mismatch, etc.
```
