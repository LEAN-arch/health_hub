# health_hub/utils/ui_visualization_helpers.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
from config import app_config # Assuming app_config is accessible
import html
import geopandas as gpd
import os

logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# --- Mapbox Token Setup ---
MAPBOX_TOKEN_SET = False
try:
    MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    # Check if token is not None, not empty, not a placeholder, and reasonably long
    if MAPBOX_ACCESS_TOKEN and MAPBOX_ACCESS_TOKEN.strip() and \
       "YOUR_MAPBOX_ACCESS_TOKEN" not in MAPBOX_ACCESS_TOKEN and \
       len(MAPBOX_ACCESS_TOKEN) > 20: # Typical Mapbox tokens are longer
        px.set_mapbox_access_token(MAPBOX_ACCESS_TOKEN)
        MAPBOX_TOKEN_SET = True
        logger.info("Mapbox access token found and set for Plotly Express.")
    else:
        if MAPBOX_ACCESS_TOKEN: # If it exists but is placeholder/short
            logger.warning("MAPBOX_ACCESS_TOKEN environment variable is a placeholder or too short. Map styles requiring a token may not work; defaulting to open styles.")
        else: # If not found at all
            logger.warning("MAPBOX_ACCESS_TOKEN environment variable not found. Map styles requiring a token may not work; defaulting to open styles.")
except Exception as e_token: # pragma: no cover
    logger.error(f"Error setting Mapbox token: {e_token}")


# utils/ui_visualization_helpers.py
# ... (other imports and code remain the same) ...

# --- Global Plotly Theme ---
def set_custom_plotly_theme():
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"'
    theme_primary_text_color = "#212529" # Dark Gray for text
    theme_secondary_text_color = "#495057" # Medium Gray
    theme_grid_color = "#e9ecef" # Light Gray for grids
    theme_border_color = "#ced4da" # For axes lines
    # Corrected: Safely access app_config properties for theme colors
    try:
        # Assuming app_config.STYLE_CSS is a dict like: {'body': {'background-color': '#f8f9fa'}}
        # This part might need adjustment based on actual structure of app_config.STYLE_CSS
        # For now, using a fallback if app_config.STYLE_CSS is not structured as expected.
        if hasattr(app_config, 'STYLE_CSS') and isinstance(app_config.STYLE_CSS, dict) and \
           'body' in app_config.STYLE_CSS and isinstance(app_config.STYLE_CSS['body'], dict) and \
           'background-color' in app_config.STYLE_CSS['body']:
            theme_paper_bg_color = app_config.STYLE_CSS['body']['background-color']
        else:
            theme_paper_bg_color = "#f8f9fa" # Default fallback
    except AttributeError: # If STYLE_CSS or its keys don't exist
         theme_paper_bg_color = "#f8f9fa" # Default fallback


    theme_plot_bg_color = "#FFFFFF"  # White plot background typically looks clean

    custom_theme = go.layout.Template()
    custom_theme.layout.font = dict(family=theme_font_family, size=12, color=theme_primary_text_color)
    custom_theme.layout.paper_bgcolor = theme_paper_bg_color
    custom_theme.layout.plot_bgcolor = theme_plot_bg_color
    
    custom_theme.layout.colorway = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1', '#fd7e14', '#20c997', '#6610f2', '#e83e8c']

    axis_common = dict(
        gridcolor=theme_grid_color,
        linecolor=theme_border_color,
        zerolinecolor=theme_grid_color, 
        zerolinewidth=1,
        title_font_size=13,
        tickfont_size=11,
        automargin=True, 
        title_standoff=15 
    )
    custom_theme.layout.xaxis = {**axis_common}
    custom_theme.layout.yaxis = {**axis_common}
    
    # MODIFIED SECTION FOR layout.title
    custom_theme.layout.title = dict(
        font=dict( # font_size must be nested within a 'font' dict
            size=18
            # 'font_weight' is not a direct Plotly schema property for layout.title.font.
            # Bolding for titles is often default or controlled by font family choice / HTML in text.
        ),
        x=0.01, 
        xanchor='left', 
        y=0.98, # As per your latest version
        yanchor='top',
        pad=dict(t=10, b=10)
    )
    # END OF MODIFIED SECTION

    custom_theme.layout.legend = dict(
        bgcolor='rgba(255,255,255,0.85)', bordercolor=theme_border_color,borderwidth=1,
        orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
        font_size=11,
        traceorder='normal' 
    )
    custom_theme.layout.margin = dict(l=70, r=30, t=80, b=70) 
    
    default_mapbox_style = app_config.MAPBOX_STYLE
    if not MAPBOX_TOKEN_SET and app_config.MAPBOX_STYLE not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        default_mapbox_style = "open-street-map" 
        logger.info(f"Plotly theme: Mapbox style '{app_config.MAPBOX_STYLE}' requires token, defaulting mapbox.style to 'open-street-map'.")
    
    custom_theme.layout.mapbox = dict(
        style=default_mapbox_style, 
        center=dict(lat=app_config.MAP_DEFAULT_CENTER_LAT, lon=app_config.MAP_DEFAULT_CENTER_LON),
        zoom=app_config.MAP_DEFAULT_ZOOM
    )
    pio.templates["custom_health_theme"] = custom_theme
    pio.templates.default = "plotly+custom_health_theme" 
    logger.info("Custom Plotly theme 'custom_health_theme' set as default.")

# set_custom_plotly_theme() # This line should remain here, it's called on import

# ... (rest of the ui_visualization_helpers.py file remains the same) ...
set_custom_plotly_theme() # Apply theme on import

# --- Styled Components ---
def render_kpi_card(title, value, icon, status="neutral", delta=None, delta_type="neutral", help_text=None, icon_is_html=False):
    # Validate status and delta_type
    valid_statuses = {"High", "Moderate", "Low", "Neutral", "Good", "Bad"} # Extended for more semantic meaning
    valid_delta_types = {"positive", "negative", "neutral"}
    
    final_status_class = status.lower() if status.lower() in valid_statuses else "neutral"
    final_delta_type = delta_type.lower() if delta_type.lower() in valid_delta_types else "neutral"

    # Add semantic class for dual meaning status (e.g. status-good status-high)
    semantic_status_class = ""
    if status.lower() == "good": semantic_status_class = "status-good"
    elif status.lower() == "bad": semantic_status_class = "status-bad"


    delta_str = str(delta) if delta is not None and str(delta).strip() else ""
    delta_html = f'<p class="kpi-delta {final_delta_type}">{html.escape(delta_str)}</p>' if delta_str else ""

    help_text_str = str(help_text) if help_text is not None and str(help_text).strip() else ""
    tooltip_html = f'title="{html.escape(help_text_str)}"' if help_text_str else ''

    icon_str = str(icon) if icon is not None else "●" # Default character icon
    icon_display = icon_str if icon_is_html else f'<span class="kpi-icon-text">{html.escape(icon_str)}</span>'

    title_str = str(title) if title is not None else "N/A"
    value_str = str(value) if value is not None else "N/A"

    # Main status class for coloring, potentially combined with semantic for specific overrides
    # e.g. class="kpi-card status-high status-good" -> CSS can target .kpi-card.status-high.status-good
    combined_status_class = f"{final_status_class} {semantic_status_class}".strip()


    html_content = f"""
<div class="kpi-card {combined_status_class}" {tooltip_html}>
    <div class="kpi-card-header">
        <div class="kpi-icon">{icon_display}</div>
        <h3 class="kpi-title">{html.escape(title_str)}</h3>
    </div>
    <div class="kpi-body">
        <p class="kpi-value">{html.escape(value_str)}</p>
        {delta_html}
    </div>
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


def render_traffic_light(message, status, details=""):
    valid_statuses_tl = {"High", "Moderate", "Low", "Neutral"}
    dot_status_class = "status-" + (status.lower() if status and status.lower() in valid_statuses_tl else "neutral")

    details_str = str(details) if details is not None and str(details).strip() else ""
    message_str = str(message) if message is not None else "Status Unavailable"
    details_html = f'<span class="traffic-light-details">{html.escape(details_str)}</span>' if details_str else ""

    html_content = f"""
<div class="traffic-light-indicator">
    <span class="traffic-light-dot {dot_status_class}"></span>
    <span class="traffic-light-message">{html.escape(message_str)}</span>
    {details_html}
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# --- Plotting Functions ---
def _create_empty_figure(title, height, message="No data available to display."):
    """Helper to create a standard empty figure."""
    fig = go.Figure()
    fig.update_layout(
        title_text=f"{title} ({message})", height=height,
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[dict(text=message, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=14))]
    )
    return fig


def plot_layered_choropleth_map(
    gdf: gpd.GeoDataFrame, value_col: str, title: str,
    id_col: str = 'zone_id', featureidkey_prefix: str = 'properties', # Set to None or empty string if IDs are at feature.id
    color_continuous_scale: str = "Blues_r", # Default to reverse Blues (darker for higher values)
    hover_cols: list = None,
    facility_gdf: gpd.GeoDataFrame = None, facility_size_col: str = None, facility_hover_name: str = None,
    facility_color: str = '#6F42C1', # Default facility color (Purple from theme)
    height: int = None, center_lat: float = None, center_lon: float = None, zoom_level: int = None,
    mapbox_style: str = None # Allow override of theme's mapbox style per plot
):
    final_height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    logger.debug(f"Plotting Choropleth: '{title}', Value Col: '{value_col}', ID Col: '{id_col}', Feature ID Prefix: '{featureidkey_prefix}'")

    error_message_map = "Map data unavailable or configuration error."
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        logger.error(f"MAP ERROR ({title}): Input 'gdf' is not a valid or non-empty GeoDataFrame.")
        return _create_empty_figure(title, final_height, error_message_map)

    active_geom_col = gdf.geometry.name
    if active_geom_col not in gdf.columns or gdf[active_geom_col].is_empty.all() or not gdf[active_geom_col].is_valid.any():
        logger.error(f"MAP ERROR ({title}): Geometry column '{active_geom_col}' missing, all empty, or no valid geometries.")
        return _create_empty_figure(title, final_height, "Invalid or empty geometries.")

    gdf_plot = gdf.copy() # Work on a copy
    if id_col not in gdf_plot.columns:
        logger.error(f"MAP ERROR ({title}): ID column '{id_col}' not found in GeoDataFrame.")
        return _create_empty_figure(title, final_height, f"ID column '{id_col}' missing.")
    if value_col not in gdf_plot.columns:
        logger.error(f"MAP ERROR ({title}): Value column '{value_col}' not found for coloring.")
        return _create_empty_figure(title, final_height, f"Value column '{value_col}' missing.")

    # Ensure value_col is numeric for color scale
    if not pd.api.types.is_numeric_dtype(gdf_plot[value_col]):
        gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors='coerce')
    if gdf_plot[value_col].isnull().all():
        logger.warning(f"MAP WARNING ({title}): All values in '{value_col}' are NaN. Map will lack color variation.")
        # To prevent error, fill NaN with a placeholder (e.g., 0) for color mapping
        gdf_plot[value_col] = gdf_plot[value_col].fillna(0)


    # GeoJSON features for choroplethmapbox must have IDs that match the 'locations' parameter.
    # If 'id_col' (e.g. 'zone_id') values are directly at feature.id, set featureidkey_prefix=None.
    # If 'id_col' values are at feature.properties.zone_id, set featureidkey_prefix='properties'.
    featureidkey_path = f"{featureidkey_prefix}.{id_col}" if featureidkey_prefix else id_col
    
    # Prepare GeoDataFrame by ensuring only valid geometries and `id_col` is string for matching
    gdf_plot[id_col] = gdf_plot[id_col].astype(str)
    gdf_for_geojson = gdf_plot[gdf_plot.geometry.is_valid & ~gdf_plot.geometry.is_empty].copy()
    if gdf_for_geojson.empty:
        logger.error(f"MAP ERROR ({title}): No valid, non-empty geometries after filtering.")
        return _create_empty_figure(title, final_height, "No valid geometries for map display.")
    
    # Forcing the id_col into properties ensures featureidkey="properties.your_id_col" works
    # This standardizes how Plotly finds the ID in the GeoJSON properties.
    # The __geo_interface__ standard should create features with a 'properties' member.
    # If `gdf_for_geojson` has `id_col` as a column, `__geo_interface__` should include it in properties.

    current_mapbox_style = mapbox_style if mapbox_style else pio.templates.default.layout.mapbox.style
    if not MAPBOX_TOKEN_SET and current_mapbox_style not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        logger.warning(f"Mapbox token not set for style '{current_mapbox_style}', defaulting to 'open-street-map' for map '{title}'.")
        current_mapbox_style = "open-street-map"

    # Set hover data carefully to avoid errors with missing columns
    hover_name_col = "name" if "name" in gdf_plot.columns else id_col
    default_hover_data_cols = [hover_name_col, value_col, 'population'] # Common useful hover data
    final_hover_data_list = hover_cols if hover_cols else default_hover_data_cols
    # Filter to ensure all hover_data columns actually exist in gdf_plot
    hover_data_for_plot = {col: True for col in final_hover_data_list if col in gdf_plot.columns and col != hover_name_col}


    fig_args = {
        "data_frame": gdf_for_geojson, # Use the GDF directly, px handles GeoJSON conversion
        "locations": id_col,          # Column in GDF matching IDs
        "featureidkey": featureidkey_path, # How to find ID in GeoJSON features. Correct from original.
        "color": value_col,
        "color_continuous_scale": color_continuous_scale,
        "opacity": 0.75,
        "hover_name": hover_name_col,
        "hover_data": hover_data_for_plot, # Use the filtered dict
        "labels": {col: col.replace('_', ' ').title() for col in [value_col] + list(hover_data_for_plot.keys())},
        "mapbox_style": current_mapbox_style,
        "center": {"lat": center_lat or app_config.MAP_DEFAULT_CENTER_LAT, "lon": center_lon or app_config.MAP_DEFAULT_CENTER_LON},
        "zoom": zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM
    }

    try:
        fig = px.choropleth_mapbox(**fig_args)
    except Exception as e_px:
        logger.error(f"MAP ERROR ({title}): Plotly Express choropleth_mapbox call failed: {e_px}", exc_info=True)
        return _create_empty_figure(title, final_height, f"Map rendering error: {e_px}")

    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        facility_plot_gdf = facility_gdf[facility_gdf.geometry.geom_type == 'Point'].copy()
        if not facility_plot_gdf.empty:
            facility_hover_text = facility_plot_gdf.get(facility_hover_name, "Facility") if facility_hover_name else "Facility"
            
            # Scale facility marker size if size_col provided
            facility_marker_size = 10 # Default size
            if facility_size_col and facility_size_col in facility_plot_gdf.columns and pd.api.types.is_numeric_dtype(facility_plot_gdf[facility_size_col]):
                sizes = pd.to_numeric(facility_plot_gdf[facility_size_col], errors='coerce').fillna(0)
                min_size_px, max_size_px = 6, 20 # Pixel size range for markers
                min_val, max_val = sizes.min(), sizes.max()
                if max_val > min_val: # Avoid division by zero if all values are same
                    facility_marker_size = min_size_px + ((sizes - min_val) * (max_size_px - min_size_px) / (max_val - min_val))
                elif sizes.notna().any(): facility_marker_size = (min_size_px + max_size_px) / 2
            
            fig.add_trace(go.Scattermapbox(
                lon=facility_plot_gdf.geometry.x, lat=facility_plot_gdf.geometry.y,
                mode='markers',
                marker=go.scattermapbox.Marker(size=facility_marker_size, sizemin=5, color=facility_color, opacity=0.9, allowoverlap=True),
                text=facility_hover_text, hoverinfo='text', name='Facilities Layer' # Simpler name for legend
            ))
    
    fig.update_layout(title_text=title, height=final_height, margin={"r":10,"t":60,"l":10,"b":10}, # Increased top margin for title
                      legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'))
    return fig


def plot_annotated_line_chart(
    data_series: pd.Series, title: str, y_axis_title: str = "Value", color: str = None,
    target_line: float = None, target_label: str = None, show_ci: bool = False,
    lower_bound_series: pd.Series = None, upper_bound_series: pd.Series = None,
    height: int = None, show_anomalies: bool = True, date_format: str = "%b %Y" # Example: "Jan 2023"
):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    fig_empty_layout = go.Layout(title_text=f"{title} (No data available)", height=final_height,
                                 xaxis={'visible': False}, yaxis={'visible': False},
                                 annotations=[dict(text="No data to display", xref="paper", yref="paper",
                                                   showarrow=False, font=dict(size=14))])
    if not isinstance(data_series, pd.Series) or data_series.empty:
        logger.warning(f"Empty or invalid data_series for line chart: {title}")
        return go.Figure(layout=fig_empty_layout)

    fig = go.Figure()
    
    # --- CORRECTED line_color ASSIGNMENT ---
    line_color = color # Use provided color if any
    if not line_color: # If no color is provided, try to get from theme
        try:
            # Access the registered template object directly
            current_template_obj = pio.templates["custom_health_theme"]
            if hasattr(current_template_obj, 'layout') and hasattr(current_template_obj.layout, 'colorway') and current_template_obj.layout.colorway:
                line_color = current_template_obj.layout.colorway[0]
            else: # Fallback if colorway not found in custom theme's layout
                logger.debug("Colorway not found in custom_health_theme.layout, trying Plotly's default theme's colorway.")
                plotly_default_template = pio.templates["plotly"] # Access the base "plotly" template
                if hasattr(plotly_default_template, 'layout') and hasattr(plotly_default_template.layout, 'colorway') and plotly_default_template.layout.colorway:
                    line_color = plotly_default_template.layout.colorway[0]
                else: # Absolute fallback
                    line_color = "#007bff" 
                    logger.warning("Could not retrieve default colorway from Plotly templates. Using hardcoded default blue for line chart.")
        except Exception as e_color:
            logger.warning(f"Error accessing template colorway for line chart '{title}': {e_color}. Using hardcoded default blue.")
            line_color = "#007bff" # Hardcoded fallback blue if all else fails
    # --- END OF CORRECTED line_color ASSIGNMENT ---


    fig.add_trace(go.Scatter(
        x=data_series.index, y=data_series.values, mode="lines+markers", name=y_axis_title,
        line=dict(color=line_color, width=2.8), marker=dict(size=7, symbol='circle'),
        customdata=data_series.values, # Access original values in hovertemplate
        hovertemplate=(f'<b>Date</b>: %{{x|{date_format}}}<br>'
                       f'<b>{y_axis_title}</b>: %{{customdata:,.2f}}' # Adjust format as needed
                       '<extra></extra>') # Hides trace name
    ))

    # Confidence Interval
    if show_ci and lower_bound_series is not None and upper_bound_series is not None and \
       not lower_bound_series.empty and not upper_bound_series.empty:
        common_idx_ci = data_series.index.intersection(lower_bound_series.index).intersection(upper_bound_series.index)
        if not common_idx_ci.empty:
            ls = pd.to_numeric(lower_bound_series.reindex(common_idx_ci), errors='coerce')
            us = pd.to_numeric(upper_bound_series.reindex(common_idx_ci), errors='coerce')
            valid_ci_mask = ls.notna() & us.notna()
            if valid_ci_mask.any():
                x_ci_plot = common_idx_ci[valid_ci_mask]
                y_upper_plot = us[valid_ci_mask]
                y_lower_plot = ls[valid_ci_mask]
                
                # Ensure fillcolor is valid by parsing the line_color
                fill_color_rgba = f"rgba({','.join(str(int(c, 16)) for c in (line_color[1:3], line_color[3:5], line_color[5:7]))},0.18)" \
                                   if line_color.startswith('#') and len(line_color) == 7 else "rgba(0,123,255,0.18)" # Default fill if color parse fails


                fig.add_trace(go.Scatter(
                    x=list(x_ci_plot) + list(x_ci_plot[::-1]), 
                    y=list(y_upper_plot.values) + list(y_lower_plot.values[::-1]), 
                    fill="toself",
                    fillcolor=fill_color_rgba, 
                    line=dict(width=0), name="Confidence Interval", hoverinfo='skip'
                ))

    # Target Line
    if target_line is not None:
        fig.add_hline(
            y=target_line, line_dash="dot", line_color="#e74c3c", line_width=1.8, # Thicker, dotted red
            annotation_text=target_label if target_label else f"Target: {target_line:,.2f}",
            annotation_position="top right", annotation_font_size=11, annotation_font_color="#c0392b"
        )

    # Anomaly Detection (using IQR method for robustness)
    if show_anomalies and len(data_series) > 10 and data_series.nunique() > 1: # Need enough data points
        q1 = data_series.quantile(0.25)
        q3 = data_series.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 1e-7: # Ensure IQR is meaningful and not zero
            anomaly_multiplier = 1.5 # Standard for mild outliers
            upper_bound = q3 + anomaly_multiplier * iqr
            lower_bound = q1 - anomaly_multiplier * iqr
            anomalies = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values, mode='markers',
                    marker=dict(color='#fd7e14', size=11, symbol='x-thin-open', line=dict(width=2.8)), # Bright orange 'x'
                    name='Potential Anomaly',
                    customdata=anomalies.values,
                    hovertemplate=(f'<b>Anomaly Date</b>: %{{x|{date_format}}}<br>'
                                   '<b>Value</b>: %{customdata:,.2f}<extra></extra>')
                ))
        else: logger.debug(f"Anomaly detection skipped for '{title}': low variance (IQR is zero or NaN) or insufficient data.")

    final_xaxis_title = data_series.index.name if data_series.index.name else "Date"
    fig.update_layout(
        title_text=title, xaxis_title=final_xaxis_title, yaxis_title=y_axis_title,
        height=final_height, hovermode="x unified", 
        legend=dict(traceorder='normal') 
    )
    return fig


def plot_bar_chart(
    df_input, x_col: str, y_col: str, title: str, color_col: str = None, barmode: str = 'group',
    orientation: str = 'v', y_axis_title: str = None, x_axis_title: str = None, height: int = None,
    text_auto: bool = True, sort_values_by: str = None, ascending: bool = True, text_format: str = ',.0f', # Default to integer display for bar values
    color_discrete_map: dict = None
):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if df_input is None or df_input.empty or x_col not in df_input.columns or y_col not in df_input.columns:
        logger.warning(f"Empty or invalid data for bar chart: {title}")
        return _create_empty_figure(title, final_height)

    df = df_input.copy()
    # Ensure y_col is numeric
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df.dropna(subset=[x_col, y_col], inplace=True) # x_col for categories, y_col for values
    if df.empty:
         logger.warning(f"Bar chart '{title}' has no valid data after cleaning '{x_col}' and '{y_col}'.")
         return _create_empty_figure(title, final_height)

    final_y_axis_title = y_axis_title if y_axis_title else y_col.replace('_', ' ').title()
    final_x_axis_title = x_axis_title if x_axis_title else x_col.replace('_', ' ').title()

    if sort_values_by and sort_values_by in df.columns:
        try: df.sort_values(by=sort_values_by, ascending=ascending, inplace=True, na_position='last',
                            key=(lambda col_data: col_data.astype(str) if not pd.api.types.is_numeric_dtype(col_data) else None))
        except Exception as e_sort: logger.warning(f"Could not sort bar chart '{title}' by '{sort_values_by}': {e_sort}. Proceeding unsorted.")
    
    # If color_col is provided, use it for legend title as well
    legend_title_text = color_col.replace('_',' ').title() if color_col and color_col in df.columns else None

    fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col,
                 barmode=barmode, orientation=orientation, height=final_height,
                 labels={y_col: final_y_axis_title, x_col: final_x_axis_title, color_col: legend_title_text if legend_title_text else ""},
                 text_auto=text_auto, color_discrete_map=color_discrete_map)

    # Common hovertemplate structure
    base_hover = f'<b>{final_x_axis_title}</b>: %{{x}}<br><b>{final_y_axis_title}</b>: %{{y:{text_format}}}'
    if color_col and color_col in df.columns:
        hover_template_str = base_hover + f'<br><b>{legend_title_text}</b>: %{{customdata[0]}}<extra></extra>'
        custom_data_for_hover = df[[color_col]] # px.bar can use this if color != color_col
    else:
        hover_template_str = base_hover + '<extra></extra>'
        custom_data_for_hover = None
        
    fig.update_traces(
        marker_line_width=0.8, marker_line_color='rgba(40,40,40,0.5)',
        textfont_size=11, textangle=0, textposition='auto' if orientation == 'v' else 'outside', cliponaxis=False,
        texttemplate=f'%{{y:{text_format}}}' if text_auto and orientation == 'v' else (f'%{{x:{text_format}}}' if text_auto and orientation == 'h' else None),
        hovertemplate = hover_template_str,
        customdata = custom_data_for_hover
    )
    fig.update_layout(yaxis_title=final_y_axis_title, xaxis_title=final_x_axis_title,
                      uniformtext_minsize=9, uniformtext_mode='hide',
                      legend_title_text=legend_title_text) # Add legend title explicitly
    
    # For horizontal bars, often good to order categories by total value if not already sorted.
    if orientation == 'h' and not sort_values_by: # Only if no explicit sort defined
        fig.update_layout(yaxis={'categoryorder':'total ascending' if ascending else 'total descending'})
    return fig


def plot_donut_chart(
    data_df_input, labels_col: str, values_col: str, title: str, height: int = None,
    color_discrete_map: dict = None, pull_segments: float = 0.03, center_text: str = None
):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 40 # Donuts may need more space for legend
    if data_df_input is None or data_df_input.empty or labels_col not in data_df_input.columns or values_col not in data_df_input.columns:
        logger.warning(f"Empty or invalid data for donut chart: {title}")
        return _create_empty_figure(title, final_height)

    df = data_df_input.copy()
    df[values_col] = pd.to_numeric(df[values_col], errors='coerce').fillna(0)
    df = df[df[values_col] > 0] # Plot only segments with positive values
    if df.empty:
        logger.warning(f"No positive values to plot for donut chart: {title}")
        return _create_empty_figure(title, final_height, "No positive data to display.")
    
    df.sort_values(by=values_col, ascending=False, inplace=True) # Sort for consistent pull effect / legend order

    # Prepare colors
    plot_colors = None
    if color_discrete_map:
        plot_colors = [color_discrete_map.get(label, pio.templates.default.layout.colorway[i % len(pio.templates.default.layout.colorway)]) 
                       for i, label in enumerate(df[labels_col])]
    
    fig = go.Figure(data=[go.Pie(
        labels=df[labels_col], values=df[values_col], hole=0.52, # Larger hole for modern look
        pull=[pull_segments if i < 3 else 0 for i in range(len(df))], # Pull top 3 segments
        textinfo='label+percent', hoverinfo='label+value+percent',
        insidetextorientation='radial', # Try radial for better fit
        marker=dict(colors=plot_colors, line=dict(color='#ffffff', width=2.2)),
        sort=False # Already sorted dataframe
    )])
    
    # Add text in the center of the donut if provided
    annotations_list = []
    if center_text:
        annotations_list.append(dict(text=center_text, x=0.5, y=0.5, font_size=18, showarrow=False, font_color=theme_primary_text_color))

    fig.update_layout(title_text=title, height=final_height, showlegend=True,
                      legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.18, traceorder="sorted"),
                      annotations=annotations_list if annotations_list else None)
    return fig


def plot_heatmap(
    matrix_df_input, title: str, height: int = None, colorscale: str = "RdBu_r", # Diverging scale good for correlations
    zmid: float = 0, text_auto: bool = True, text_format: str = ".2f", show_colorbar: bool = True
):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 100 # Heatmaps can be tall
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty:
        logger.error(f"Invalid input for heatmap: {title}. Must be a non-empty DataFrame.")
        return _create_empty_figure(title, final_height, "Invalid data for Heatmap.")

    # Attempt to convert all data to numeric, crucial for heatmap Z values.
    df_numeric = matrix_df_input.copy()
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # If all values are NaN after conversion, there's no numeric data to plot.
    if df_numeric.isnull().all().all() and not matrix_df_input.empty: # matrix_df_input check handles if original was just NaNs
         logger.error(f"Matrix for heatmap '{title}' contains only non-numeric or unconvertible data.")
         return _create_empty_figure(title, final_height, "All data non-numeric or unconvertible.")
    
    # Fill NaNs: For heatmaps, often replace with 0 or a value indicating missing data.
    # Here, 0 is a common choice, but could make `zmid=0` less meaningful if 0 is a valid data point.
    # A dedicated color for NaNs could be better (`hovertemplate` can show "N/A"). For simplicity, fill with 0.
    df_plot = df_numeric.fillna(0)

    z_values = df_plot.values
    text_values_for_template = np.around(z_values, decimals=int(text_format[-2]) if text_format.endswith('f') else 2)

    fig = go.Figure(data=go.Heatmap(
        z=z_values, x=df_plot.columns.tolist(), y=df_plot.index.tolist(),
        colorscale=colorscale, zmid=zmid if pd.Series(z_values.flatten()).min() < 0 and pd.Series(z_values.flatten()).max() > 0 else None, # Only use zmid if data is diverging
        text=text_values_for_template if text_auto else None,
        texttemplate=f"%{{text:{text_format}}}" if text_auto else None,
        hoverongaps=False, xgap=1.8, ygap=1.8, # Slightly larger gaps
        colorbar=dict(thickness=20, len=0.9, tickfont_size=10, title_side="right", outlinewidth=1, outlinecolor=theme_border_color) if show_colorbar else None
    ))
    
    # Determine if X-axis labels need rotation based on number and length
    rotate_x_labels = -40 if len(df_plot.columns) > 8 or max(len(str(c)) for c in df_plot.columns) > 10 else 0
    
    fig.update_layout(
        title_text=title, height=final_height,
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_tickangle=rotate_x_labels,
        yaxis_autorange='reversed', # Typical for heatmaps/matrices
        plot_bgcolor='rgba(0,0,0,0)' # Transparent bg for heatmap itself
    )
    return fig
