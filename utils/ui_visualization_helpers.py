# utils/ui_visualization_helpers.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
from config import app_config
import html
import geopandas as gpd # Ensure gpd is imported here too
import os # For Mapbox token

# Configure logging
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Mapbox Token Setup (CRUCIAL) ---
MAPBOX_TOKEN_SET = False
try:
    MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    if MAPBOX_ACCESS_TOKEN and "YOUR_MAPBOX_ACCESS_TOKEN" not in MAPBOX_ACCESS_TOKEN: # Avoid placeholder
        px.set_mapbox_access_token(MAPBOX_ACCESS_TOKEN)
        MAPBOX_TOKEN_SET = True
        logger.info("Mapbox access token found and set for Plotly Express.")
    else:
        logger.warning("MAPBOX_ACCESS_TOKEN environment variable not found or is placeholder. Map styles other than 'open-street-map' may not work.")
except Exception as e_token: # pragma: no cover
    logger.error(f"Error setting Mapbox token: {e_token}")


# --- Global Plotly Theme ---
def set_custom_plotly_theme():
    custom_theme = go.layout.Template()
    custom_theme.layout.font = dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size=12, color="#374151")
    custom_theme.layout.paper_bgcolor = "#FFFFFF"
    custom_theme.layout.plot_bgcolor = "#FFFFFF"
    custom_theme.layout.colorway = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#6366F1', '#8B5CF6']
    custom_theme.layout.xaxis = dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#E5E7EB", title_font_size=13, tickfont_size=11, automargin=True)
    custom_theme.layout.yaxis = dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#E5E7EB", title_font_size=13, tickfont_size=11, automargin=True)
    custom_theme.layout.title = dict(font_size=16, font_weight='bold', x=0.02, xanchor='left', y=0.95, yanchor='top')
    custom_theme.layout.legend = dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#E5E7EB', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    custom_theme.layout.margin = dict(l=60, r=40, t=70, b=60)
    pio.templates["custom_health_theme"] = custom_theme
    pio.templates.default = "plotly+custom_health_theme"
set_custom_plotly_theme()

# --- Styled Components ---
def render_kpi_card(title, value, icon, status=None, delta=None, delta_type="neutral", help_text=None, icon_is_html=False):
    status_class_map = {"High": "status-high", "Moderate": "status-moderate", "Low": "status-low"}
    status_final_class = status_class_map.get(status, "")
    delta_html = f'<p class="kpi-delta {delta_type}">{html.escape(str(delta))}</p>' if delta else ""
    tooltip_html = f'title="{html.escape(str(help_text))}"' if help_text else ''
    icon_display = str(icon) if icon_is_html else html.escape(str(icon))
    html_content = f"""<div class="kpi-card {status_final_class}" {tooltip_html}> <div class="kpi-card-header"> <span class="kpi-icon">{icon_display}</span> <h3 class="kpi-title">{html.escape(str(title))}</h3> </div> <div> <p class="kpi-value">{html.escape(str(value))}</p> {delta_html} </div></div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message, status, details=""):
    status_class_map = {"High": "status-high", "Moderate": "status-moderate", "Low": "status-low", "Neutral": "status-neutral"}
    dot_status_class = status_class_map.get(status, "status-neutral")
    details_html = f'<span class="traffic-light-details">{html.escape(str(details))}</span>' if details else ""
    html_content = f"""<div class="traffic-light-indicator"> <span class="traffic-light-dot {dot_status_class}"></span> <span class="traffic-light-message">{html.escape(str(message))}</span> {details_html}</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)

# --- Plotting Functions ---
def plot_layered_choropleth_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    id_col: str = 'zone_id',
    featureidkey_prop: str = 'zone_id',
    color_continuous_scale: str = "OrRd",
    hover_cols: list = None,
    facility_gdf: gpd.GeoDataFrame = None,
    facility_size_col: str = None,
    facility_hover_name: str = None,
    height: int = None,
    center_lat: float = None,
    center_lon: float = None,
    zoom_level: int = None,
    mapbox_style: str = "carto-positron"
):
    height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    logger.debug(f"--- Plotting Choropleth: {title} ---")
    logger.debug(f"Input GDF type: {type(gdf)}, shape: {gdf.shape if gdf is not None else 'None'}")
    logger.debug(f"Params: value_col='{value_col}', id_col='{id_col}', featureidkey_prop='{featureidkey_prop}'")
    logger.debug(f"Mapbox style: '{mapbox_style}', Token set: {MAPBOX_TOKEN_SET}")

    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        logger.error(f"MAP_ERROR ({title}): Input 'gdf' is not a valid or non-empty GeoDataFrame.")
        return go.Figure().update_layout(title_text=f"{title} (No geographic data)", height=height, annotations=[dict(text="Map data unavailable", xref="paper", yref="paper", showarrow=False)])
    active_geom_col = gdf.geometry.name
    if active_geom_col not in gdf.columns:
        logger.error(f"MAP_ERROR ({title}): Active geometry column '{active_geom_col}' not found. Columns: {gdf.columns.tolist()}")
        return go.Figure().update_layout(title_text=f"{title} (Missing geometry column)", height=height, annotations=[dict(text="Map config error", xref="paper", yref="paper", showarrow=False)])
    if gdf.geometry.is_empty.all():
        logger.error(f"MAP_ERROR ({title}): All geometries in gdf are empty.")
        return go.Figure().update_layout(title_text=f"{title} (All Geometries Empty)", height=height, annotations=[dict(text="Map geometry error", xref="paper", yref="paper", showarrow=False)])
    
    gdf_plot = gdf.copy()
    if id_col not in gdf_plot.columns:
        logger.error(f"MAP_ERROR ({title}): ID column '{id_col}' not found in gdf_plot.")
        return go.Figure().update_layout(title_text=f"{title} (Missing ID column: {id_col})", height=height, annotations=[dict(text="Map config error", xref="paper", yref="paper", showarrow=False)])
    if value_col not in gdf_plot.columns:
        logger.error(f"MAP_ERROR ({title}): Value column '{value_col}' not found in gdf_plot.")
        return go.Figure().update_layout(title_text=f"{title} (Missing value column: {value_col})", height=height, annotations=[dict(text="Map config error", xref="paper", yref="paper", showarrow=False)])

    if not pd.api.types.is_numeric_dtype(gdf_plot[value_col]):
        logger.info(f"Value column '{value_col}' for '{title}' is not numeric. Attempting conversion.")
        gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors='coerce')
    if gdf_plot[value_col].isnull().all():
        logger.warning(f"All values in '{value_col}' for '{title}' are NaN. Map may show shapes but no color variation.")

    gdf_for_geojson = gdf_plot[gdf_plot.geometry.is_valid & ~gdf_plot.geometry.is_empty]
    if gdf_for_geojson.empty:
        logger.error(f"MAP_ERROR ({title}): No valid, non-empty geometries after filtering.")
        return go.Figure().update_layout(title_text=f"{title} (No valid geometries)", height=height, annotations=[dict(text="Map geometry processing error", xref="paper", yref="paper", showarrow=False)])
    try:
        geojson_interface = gdf_for_geojson.geometry.__geo_interface__
        if not geojson_interface or not geojson_interface.get("features"):
            logger.error(f"Failed to generate GeoJSON features for '{title}'.")
            return go.Figure().update_layout(title_text=f"{title} (GeoJSON generation error)", height=height)
        if geojson_interface.get('features'): logger.debug(f"GeoJSON Feature 0 Props: {geojson_interface['features'][0]['properties'].get(featureidkey_prop)}")
        logger.debug(f"GDF ID samples: {gdf_plot[id_col].unique()[:3]}")
    except Exception as e_geojson: # pragma: no cover
        logger.error(f"MAP_ERROR ({title}): Error creating GeoJSON interface: {e_geojson}", exc_info=True)
        return go.Figure().update_layout(title_text=f"{title} (GeoJSON interface error)", height=height)

    default_hover_cols = ['name', 'population', value_col]; final_hover_cols = [col for col in (hover_cols if hover_cols else default_hover_cols) if col in gdf_plot.columns]
    
    current_mapbox_style = mapbox_style
    if not MAPBOX_TOKEN_SET and mapbox_style not in ["open-street-map", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        logger.warning(f"Mapbox token not set for style '{mapbox_style}', defaulting to 'open-street-map' for '{title}'.")
        current_mapbox_style = "open-street-map"

    px_func_args = {"mapbox_style": current_mapbox_style}
    if center_lat is not None and center_lon is not None:
        px_func_args["center"] = {"lat": center_lat, "lon": center_lon}
        px_func_args["zoom"] = zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM
    elif zoom_level is not None:
        px_func_args["zoom"] = zoom_level
    else: px_func_args["fitbounds"] = "locations"
    
    logger.info(f"Creating choropleth_mapbox for '{title}' with args: {px_func_args}")
    try:
        plot_args = {"geojson": geojson_interface, "locations": gdf_plot[id_col], "featureidkey": f"properties.{featureidkey_prop}", "color": value_col, "color_continuous_scale": color_continuous_scale, "opacity": 0.75, "hover_name": "name" if "name" in gdf_plot.columns else id_col, "hover_data": final_hover_cols, "labels": {col: col.replace('_',' ').title() for col in [value_col] + final_hover_cols}}
        plot_args.update(px_func_args)
        fig = px.choropleth_mapbox(gdf_plot, **plot_args)
    except Exception as e_px: # pragma: no cover
        logger.error(f"MAP_ERROR ({title}): px.choropleth_mapbox failed: {e_px}", exc_info=True)
        st.error(f"Map rendering error for '{title}'. Details: {e_px}")
        return go.Figure().update_layout(title_text=f"{title} (Map Rendering Error)", height=height)

    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        # ... (Facility marker logic as previously provided - unchanged from last complete file) ...
        facility_gdf_points = facility_gdf[facility_gdf.geometry.geom_type == 'Point'].copy();
        if not facility_gdf_points.empty:
            size_data = 8; hover_text_data = "Facility"; 
            if facility_size_col and facility_size_col in facility_gdf_points.columns and pd.api.types.is_numeric_dtype(facility_gdf_points[facility_size_col]):
                min_s, max_s_px = 4, 15; s_min_val, s_max_val = facility_gdf_points[facility_size_col].min(), facility_gdf_points[facility_size_col].max()
                if pd.notna(s_min_val) and pd.notna(s_max_val) and s_max_val > s_min_val : range_s = s_max_val - s_min_val; size_data_series = min_s + ((facility_gdf_points[facility_size_col] - s_min_val) * (max_s_px - min_s) / range_s); size_data = size_data_series.fillna(8) if isinstance(size_data_series, pd.Series) else size_data_series
                elif pd.notna(s_max_val) and s_max_val == s_min_val: size_data = (min_s + max_s_px) / 2
            if facility_hover_name and facility_hover_name in facility_gdf_points.columns: hover_text_data = facility_gdf_points[facility_hover_name]
            fig.add_trace(go.Scattermapbox(lon=facility_gdf_points.geometry.x, lat=facility_gdf_points.geometry.y, mode='markers', marker=go.scattermapbox.Marker(size=size_data, sizemin=4, color='#1E3A8A', opacity=0.85, allowoverlap=True), text=hover_text_data, hoverinfo='text', name='Health Facilities'))
        else: logger.debug(f"Facility GDF for '{title}' has no Point geometries.")

    fig.update_layout(title_text=title, height=height, margin={"r":5,"t":45,"l":5,"b":5}, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.6)'))
    logger.info(f"Successfully configured choropleth map: {title}")
    return fig


def plot_annotated_line_chart(data_series, title, y_axis_title="Value", color=None,
                              target_line=None, target_label=None, show_ci=False,
                              lower_bound_series=None, upper_bound_series=None,
                              height=None, show_anomalies=True):
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty:
        logger.warning(f"Empty or invalid data_series for line chart: {title}")
        fig = go.Figure()
        fig.update_layout(title_text=f"{title} (No data available)", height=height,
                          xaxis={'visible': False}, yaxis={'visible': False},
                          annotations=[dict(text="No data to display", xref="paper", yref="paper",
                                            showarrow=False, font=dict(size=14))])
        return fig

    fig = go.Figure()
    # Ensure theme is applied if pio.templates.default was set
    line_color = color if color else pio.templates.get(pio.templates.default, pio.templates["plotly"]).layout.colorway[0]


    fig.add_trace(go.Scatter(
        x=data_series.index, y=data_series.values, mode="lines+markers", name=y_axis_title,
        line=dict(color=line_color, width=2.5), marker=dict(size=6, symbol='circle-open'),
        hoverinfo='x+y', customdata=[y_axis_title]*len(data_series),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>'+y_axis_title+'</b>: %{y:,.2f}<extra></extra>' # Formatted hover
    ))

    if show_ci and lower_bound_series is not None and upper_bound_series is not None and \
       not lower_bound_series.empty and not upper_bound_series.empty:
        # Ensure indices align and data is numeric
        combined_idx = data_series.index.intersection(lower_bound_series.index).intersection(upper_bound_series.index)
        if not combined_idx.empty:
            ls = pd.to_numeric(lower_bound_series.reindex(combined_idx), errors='coerce')
            us = pd.to_numeric(upper_bound_series.reindex(combined_idx), errors='coerce')
            valid_ci_idx = ls.notna() & us.notna()
            if valid_ci_idx.any():
                common_index_ci = combined_idx[valid_ci_idx]
                ls_ci = ls[valid_ci_idx]
                us_ci = us[valid_ci_idx]
                fig.add_trace(go.Scatter(
                    x=list(common_index_ci) + list(common_index_ci[::-1]),
                    y=list(us_ci.values) + list(ls_ci.values[::-1]),
                    fill="toself",
                    fillcolor=f"rgba({','.join(str(int(c, 16)) for c in (line_color[1:3], line_color[3:5], line_color[5:7]))},0.15)",
                    line=dict(width=0), name="95% CI", hoverinfo='skip'
                ))

    if target_line is not None:
        fig.add_hline(
            y=target_line, line_dash="dash", line_color="#EF4444", # Red
            annotation_text=target_label if target_label else f"Target: {target_line}",
            annotation_position="bottom right", annotation_font_size=10, annotation_font_color="#EF4444"
        )

    if show_anomalies and len(data_series) > 5 and data_series.nunique() > 1: # Avoid on constant series
        q95 = data_series.quantile(0.95)
        q05 = data_series.quantile(0.05)
        std_dev = data_series.std()
        mean_val = data_series.mean()
        if pd.notna(std_dev) and pd.notna(mean_val) and std_dev > 1e-6: # Check std_dev is not zero or too small
            upper_anomaly_thresh = max(q95, mean_val + 2*std_dev)
            lower_anomaly_thresh = min(q05, mean_val - 2*std_dev)
            anomalies = data_series[(data_series > upper_anomaly_thresh) | (data_series < lower_anomaly_thresh)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values, mode='markers',
                    marker=dict(color='#D90429', size=10, symbol='x-thin-open', line=dict(width=2)),
                    name='Potential Anomaly', hoverinfo='x+y+text',
                    text=[f"Anomaly ({val:,.2f})" for val in anomalies.values] # Formatted text
                ))
        else:
            logger.debug(f"Anomaly detection skipped for '{title}': low variance or insufficient data.")

    fig.update_layout(
        title_text=title,
        xaxis_title=data_series.index.name if data_series.index.name else "Date",
        yaxis_title=y_axis_title,
        height=height,
        hovermode="x unified", # Shows all traces for a given x
        legend=dict(traceorder='normal', itemclick="toggleothers", itemdoubleclick="toggle")
    )
    return fig


def plot_bar_chart(df, x_col, y_col, title, color_col=None, barmode='group',
                   orientation='v', y_axis_title=None, x_axis_title=None, height=None,
                   text_auto=True, sort_values_by=None, ascending=True):
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Empty or invalid data for bar chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height, xaxis={'visible': False}, yaxis={'visible': False},
                                          annotations=[dict(text="No data to display", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    y_axis_title_final = y_axis_title if y_axis_title else y_col.replace('_', ' ').title()
    x_axis_title_final = x_axis_title if x_axis_title else x_col.replace('_', ' ').title()

    df_to_plot = df.copy()
    if y_col in df_to_plot.columns: # Ensure y_col exists before trying to convert
         df_to_plot[y_col] = pd.to_numeric(df_to_plot[y_col], errors='coerce') # Ensure y is numeric for sum/mean by color_col

    if sort_values_by and sort_values_by in df_to_plot.columns:
        try:
            # Ensure the sort column is numeric if we expect numeric sorting
            if pd.api.types.is_numeric_dtype(df_to_plot[sort_values_by]):
                df_to_plot.sort_values(by=sort_values_by, ascending=ascending, inplace=True, na_position='last')
            else: # Lexicographical sort for non-numeric
                df_to_plot.sort_values(by=sort_values_by, ascending=ascending, inplace=True, na_position='last', key=lambda col: col.astype(str))
        except Exception as e_sort: # pragma: no cover
            logger.warning(f"Could not sort bar chart '{title}' by '{sort_values_by}': {e_sort}. Proceeding unsorted.")

    fig = px.bar(df_to_plot, x=x_col, y=y_col, title=title, color=color_col,
                 barmode=barmode, orientation=orientation, height=height,
                 labels={y_col: y_axis_title_final, x_col: x_axis_title_final},
                 text_auto=text_auto)

    fig.update_traces(marker_line_width=0.8, marker_line_color='rgba(50,50,50,0.6)',
                      textfont_size=10, textangle=0, textposition='outside', cliponaxis=False)
    fig.update_layout(yaxis_title=y_axis_title_final, xaxis_title=x_axis_title_final, uniformtext_minsize=8, uniformtext_mode='hide')

    return fig


def plot_donut_chart(data_df, labels_col, values_col, title, height=None):
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 20 # Donuts often need slightly more height
    if data_df.empty or labels_col not in data_df.columns or values_col not in data_df.columns:
        logger.warning(f"Empty or invalid data for donut chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height, xaxis={'visible': False}, yaxis={'visible': False},
                                          annotations=[dict(text="No data to display", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    # Ensure values_col is numeric
    df_plot_donut = data_df.copy()
    df_plot_donut[values_col] = pd.to_numeric(df_plot_donut[values_col], errors='coerce').fillna(0)


    fig = go.Figure(data=[go.Pie(
        labels=df_plot_donut[labels_col], values=df_plot_donut[values_col], hole=0.45,
        pull=[0.02] * len(df_plot_donut), textinfo='label+percent', hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#ffffff', width=1.5))
    )])

    fig.update_layout(title_text=title, height=height, showlegend=True,
                      legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="right", x=1.1))
    return fig


def plot_heatmap(matrix_df, title, height=None, colorscale="RdBu_r", zmid=0):
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 70

    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty: # pragma: no cover
        logger.error(f"Invalid input for heatmap: {title}. Must be a non-empty DataFrame.")
        return go.Figure().update_layout(title_text=f"{title} (No data or invalid data)", height=height,
                                          annotations=[dict(text="Invalid data for Heatmap", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    try:
        # Attempt to convert all data to numeric. If a column cannot be fully converted
        # (i.e., strings that aren't numbers remain), it might indicate an issue.
        numeric_matrix_df = matrix_df.copy()
        for col in numeric_matrix_df.columns: # Convert column by column
            numeric_matrix_df[col] = pd.to_numeric(numeric_matrix_df[col], errors='coerce')
        
        # If, after attempting conversion, there are still non-numeric dtypes (excluding object if all are NaN)
        # or if all values in the matrix became NaN due to conversion issues.
        if numeric_matrix_df.isnull().all().all() and not matrix_df.empty:
             logger.error(f"Matrix for heatmap '{title}' became all NaNs after numeric conversion.")
             return go.Figure().update_layout(title_text=f"{title} (All data non-numeric or unconvertible)", height=height,
                                              annotations=[dict(text="Non-numeric data in Heatmap", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
        
        numeric_matrix_df = numeric_matrix_df.fillna(0) # Fill remaining NaNs (e.g., from original data) with 0 for plotting

    except Exception as e: # pragma: no cover
        logger.error(f"Error converting matrix to numeric for heatmap '{title}': {e}", exc_info=True)
        return go.Figure().update_layout(title_text=f"{title} (Data processing error for heatmap)", height=height,
                                          annotations=[dict(text="Error processing Heatmap data", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    zmid_val = None
    if not numeric_matrix_df.empty:
        # Calculate min/max on the actual numeric values present
        # This handles cases where entire rows/cols might be NaN before fillna(0)
        matrix_for_min_max = numeric_matrix_df.select_dtypes(include=np.number)
        if not matrix_for_min_max.empty and matrix_for_min_max.notna().any().any(): # Ensure there's some non-NaN data
            min_val = matrix_for_min_max.min().min() # Min of column minimums
            max_val = matrix_for_min_max.max().max() # Max of column maximums
            if pd.notna(min_val) and pd.notna(max_val) and min_val < 0 and max_val > 0:
                zmid_val = zmid

    fig = go.Figure(data=go.Heatmap(
        z=numeric_matrix_df.values if not numeric_matrix_df.empty else [[]],
        x=numeric_matrix_df.columns.tolist(),
        y=numeric_matrix_df.index.tolist(),
        colorscale=colorscale,
        zmid=zmid_val,
        text=np.around(numeric_matrix_df.values, decimals=2) if not numeric_matrix_df.empty else None,
        texttemplate="%{text}" if not numeric_matrix_df.empty else "",
        hoverongaps=False, xgap=1, ygap=1,
        colorbar=dict(thickness=15, len=0.75, tickfont_size=10)
    ))

    fig.update_layout(title_text=title, height=height, xaxis_showgrid=False, yaxis_showgrid=False,
                      xaxis_tickangle=-30 if len(numeric_matrix_df.columns) > 5 else 0, yaxis_autorange='reversed')
    return fig
