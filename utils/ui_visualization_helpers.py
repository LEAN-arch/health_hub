# utils/ui_visualization_helpers.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
from config import app_config # For default plot heights, etc.

# Configure logging
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO), 
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Global Plotly Theme ---
def set_custom_plotly_theme():
    # (This function remains identical to the one in your "Ultimate Dashboard" ui_visualization_helpers.py)
    # ... (ensure it's defined here as provided previously) ...
    custom_theme = go.layout.Template()
    custom_theme.layout.font = dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size=12, color="#374151")
    custom_theme.layout.paper_bgcolor = "#FFFFFF"
    custom_theme.layout.plot_bgcolor = "#FFFFFF"
    custom_theme.layout.colorway = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#6366F1', '#8B5CF6']
    custom_theme.layout.xaxis = dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#E5E7EB", title_font_size=13, tickfont_size=11, automargin=True)
    custom_theme.layout.yaxis = dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#E5E7EB", title_font_size=13, tickfont_size=11, automargin=True)
    custom_theme.layout.title = dict(font_size=16, font_weight='bold', x=0.02, xanchor='left', y=0.95, yanchor='top') # Make titles slightly more prominent
    custom_theme.layout.legend = dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#E5E7EB', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    custom_theme.layout.margin = dict(l=60, r=40, t=70, b=60) # Adjusted margins

    pio.templates["custom_health_theme"] = custom_theme
    pio.templates.default = "plotly+custom_health_theme" # Apply globally
    logger.info("Custom Plotly theme 'custom_health_theme' set as default.")

set_custom_plotly_theme() # Apply theme when module is imported

# --- Styled Components (HTML/CSS via st.markdown) ---
def render_kpi_card(title, value, icon, status=None, delta=None, delta_type="neutral", help_text=None):
    # (This function remains identical to the one in your "Ultimate Dashboard" ui_visualization_helpers.py)
    # ... (ensure it's defined here as provided previously) ...
    status_class_map = {"High": "status-high", "Moderate": "status-moderate", "Low": "status-low"}
    status_final_class = status_class_map.get(status, "")

    delta_html = f'<p class="kpi-delta {delta_type}">{delta}</p>' if delta else ""
    tooltip_html = f'title="{st.markdown.escape(help_text)}"' if help_text else '' # Escape help_text

    html_content = f"""
    <div class="kpi-card {status_final_class}" {tooltip_html}>
        <div class="kpi-card-header">
            <span class="kpi-icon">{icon}</span>
            <h3 class="kpi-title">{st.markdown.escape(title)}</h3>
        </div>
        <div>
            <p class="kpi-value">{st.markdown.escape(str(value))}</p>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message, status, details=""):
    # (This function remains identical to the one in your "Ultimate Dashboard" ui_visualization_helpers.py)
    # ... (ensure it's defined here as provided previously) ...
    status_class_map = {"High": "status-high", "Moderate": "status-moderate", "Low": "status-low", "Neutral": "status-neutral"}
    dot_status_class = status_class_map.get(status, "status-neutral")
    
    details_html = f'<span class="traffic-light-details">{st.markdown.escape(details)}</span>' if details else ""

    html_content = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {dot_status_class}"></span>
        <span class="traffic-light-message">{st.markdown.escape(message)}</span>
        {details_html}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


# --- Plotting Functions ---
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
    # Use a color from the theme's colorway if not specified
    line_color = color if color else pio.templates[pio.templates.default].layout.colorway[0]

    fig.add_trace(go.Scatter(
        x=data_series.index, y=data_series.values, mode="lines+markers", name=y_axis_title,
        line=dict(color=line_color, width=2.5), marker=dict(size=6, symbol='circle-open'),
        hoverinfo='x+y', customdata=[y_axis_title]*len(data_series), # For unified hover
        hovertemplate='<b>Date</b>: %{x}<br><b>'+y_axis_title+'</b>: %{y}<extra></extra>'
    ))

    if show_ci and lower_bound_series is not None and upper_bound_series is not None and \
       not lower_bound_series.empty and not upper_bound_series.empty:
        # Ensure indices align
        common_index = data_series.index.intersection(lower_bound_series.index).intersection(upper_bound_series.index)
        if not common_index.empty:
            ls = lower_bound_series.reindex(common_index)
            us = upper_bound_series.reindex(common_index)
            fig.add_trace(go.Scatter(
                x=list(common_index) + list(common_index[::-1]),
                y=list(us.values) + list(ls.values[::-1]),
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(c, 16)) for c in (line_color[1:3], line_color[3:5], line_color[5:7]))},0.15)",
                line=dict(width=0), name="95% CI", hoverinfo='skip'
            ))

    if target_line is not None:
        fig.add_hline(
            y=target_line, line_dash="dash", line_color="#EF4444", # Red-500
            annotation_text=target_label if target_label else f"Target: {target_line}",
            annotation_position="bottom right", annotation_font_size=10, annotation_font_color="#EF4444"
        )
    
    if show_anomalies and len(data_series) > 5: # Basic anomaly detection (example)
        q95 = data_series.quantile(0.95)
        q05 = data_series.quantile(0.05)
        # Consider std dev based anomalies too: mean +/- 2*std
        std_dev = data_series.std()
        mean_val = data_series.mean()
        upper_anomaly_thresh = max(q95, mean_val + 2*std_dev)
        lower_anomaly_thresh = min(q05, mean_val - 2*std_dev)

        anomalies = data_series[(data_series > upper_anomaly_thresh) | (data_series < lower_anomaly_thresh)]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies.values, mode='markers',
                marker=dict(color='#D90429', size=10, symbol='x-thin-open', line=dict(width=2)), # More visible anomaly marker
                name='Potential Anomaly', hoverinfo='x+y+text',
                text=[f"Anomaly ({val:.2f})" for val in anomalies.values]
            ))

    fig.update_layout(
        title_text=title,
        xaxis_title=data_series.index.name if data_series.index.name else "Date",
        yaxis_title=y_axis_title,
        height=height,
        hovermode="x unified",
        legend=dict(traceorder='normal', itemclick="toggleothers", itemdoubleclick="toggle")
    )
    logger.info(f"Rendered line chart: {title}")
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

    if sort_values_by:
        df_sorted = df.sort_values(by=sort_values_by, ascending=ascending).copy()
    else:
        df_sorted = df.copy()

    fig = px.bar(df_sorted, x=x_col, y=y_col, title=title, color=color_col,
                 barmode=barmode, orientation=orientation, height=height,
                 labels={y_col: y_axis_title_final, x_col: x_axis_title_final},
                 text_auto=text_auto) # Use Plotly's text_auto
    
    fig.update_traces(marker_line_width=0.8, marker_line_color='rgba(50,50,50,0.6)',
                      textfont_size=10, textangle=0, textposition='outside', cliponaxis=False)
    fig.update_layout(yaxis_title=y_axis_title_final, xaxis_title=x_axis_title_final, uniformtext_minsize=8, uniformtext_mode='hide')
    
    logger.info(f"Rendered bar chart: {title}")
    return fig


def plot_donut_chart(data_df, labels_col, values_col, title, height=None):
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 20 # Donuts often need a bit more height
    if data_df.empty or labels_col not in data_df.columns or values_col not in data_df.columns:
        logger.warning(f"Empty or invalid data for donut chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height, xaxis={'visible': False}, yaxis={'visible': False},
                                          annotations=[dict(text="No data to display", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    fig = go.Figure(data=[go.Pie(
        labels=data_df[labels_col], values=data_df[values_col], hole=0.45,
        pull=[0.02] * len(data_df), textinfo='label+percent', hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#ffffff', width=1.5)) # Using theme colors by default
    )])
    
    fig.update_layout(title_text=title, height=height, showlegend=True, 
                      legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="right", x=1.1)) # Vertical legend for donuts
    logger.info(f"Rendered donut chart: {title}")
    return fig


def plot_heatmap(matrix_df, title, height=None, colorscale="RdBu_r", zmid=0): # RdBu_r for diverging, zmid=0 for correlations
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 70 # Heatmaps can be taller
    
    # Basic validation
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        logger.error(f"Invalid input for heatmap: {title}. Must be a non-empty DataFrame.")
        return go.Figure().update_layout(title_text=f"{title} (No data or invalid data)", height=height,
                                          annotations=[dict(text="Invalid data for Heatmap", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
    
    # Attempt to convert to numeric, crucial for heatmap
    try:
        numeric_matrix_df = matrix_df.apply(pd.to_numeric, errors='coerce')
        if numeric_matrix_df.isnull().values.any() and not matrix_df.select_dtypes(include=np.number).equals(matrix_df): # Check if original was already numeric
            # If coercion resulted in NaNs AND original had non-numeric strings
            logger.error(f"Matrix for heatmap '{title}' contains non-numeric values that could not be converted.")
            return go.Figure().update_layout(title_text=f"{title} (Contains non-convertible non-numeric data)", height=height,
                                              annotations=[dict(text="Non-numeric data in Heatmap", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
    except Exception as e:
        logger.error(f"Error converting matrix to numeric for heatmap '{title}': {e}", exc_info=True)
        return go.Figure().update_layout(title_text=f"{title} (Data processing error for heatmap)", height=height,
                                          annotations=[dict(text="Error processing Heatmap data", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    fig = go.Figure(data=go.Heatmap(
        z=numeric_matrix_df.values, x=numeric_matrix_df.columns.tolist(), y=numeric_matrix_df.index.tolist(),
        colorscale=colorscale, zmid=zmid if (numeric_matrix_df.min().min() < 0 and numeric_matrix_df.max().max() > 0) else None, # Apply zmid if data is diverging
        text=np.around(numeric_matrix_df.values, decimals=2), texttemplate="%{text}",
        hoverongaps=False, xgap=1, ygap=1,
        colorbar=dict(thickness=15, len=0.75, tickfont_size=10)
    ))
    
    fig.update_layout(title_text=title, height=height, xaxis_showgrid=False, yaxis_showgrid=False, 
                      xaxis_tickangle=-30 if len(numeric_matrix_df.columns) > 5 else 0, yaxis_autorange='reversed') # Common for heatmaps
    logger.info(f"Rendered heatmap: {title}")
    return fig


def plot_layered_choropleth_map(gdf, value_col, title, 
                                 id_col='zone_id', featureidkey_prop='zone_id', 
                                 color_continuous_scale="OrRd", hover_cols=None, 
                                 facility_gdf=None, facility_size_col=None, facility_hover_name=None,
                                 height=None, center_lat=None, center_lon=None, zoom_level=None):
    height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    zoom_level = zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM

    if gdf is None or gdf.empty or value_col not in gdf.columns or id_col not in gdf.columns:
        logger.warning(f"Invalid GeoDataFrame or missing columns for choropleth map: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No geographic data or required metric missing)", height=height,
                                          annotations=[dict(text="Map data unavailable", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    # Ensure the value_col is numeric for color scale
    if not pd.api.types.is_numeric_dtype(gdf[value_col]):
        logger.warning(f"Value column '{value_col}' for choropleth map '{title}' is not numeric. Attempting conversion or skipping.")
        # Optionally, try to convert or raise error, for now, Plotly might handle or error out.
        # gdf[value_col] = pd.to_numeric(gdf[value_col], errors='coerce')
        # gdf.dropna(subset=[value_col], inplace=True) # if conversion leads to NaN
        # if gdf.empty: return go.Figure()... (as above)

    default_hover_cols = ['name', 'population', value_col] # Sensible defaults
    final_hover_cols = [col for col in (hover_cols if hover_cols else default_hover_cols) if col in gdf.columns]

    fig = px.choropleth_mapbox(
        gdf, geojson=gdf.geometry.__geo_interface__, locations=gdf[id_col],
        featureidkey=f"properties.{featureidkey_prop}", color=value_col,
        color_continuous_scale=color_continuous_scale, mapbox_style="carto-positron",
        opacity=0.75, hover_name="name" if "name" in gdf.columns else id_col,
        hover_data=final_hover_cols,
        labels={value_col: value_col.replace('_',' ').title()} # Auto-label for color bar
    )

    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        facility_gdf_points = facility_gdf[facility_gdf.geometry.geom_type == 'Point']
        if not facility_gdf_points.empty:
            size_data = facility_gdf_points[facility_size_col] if facility_size_col and facility_size_col in facility_gdf_points.columns else 8
            hover_text_data = facility_gdf_points[facility_hover_name] if facility_hover_name and facility_hover_name in facility_gdf_points.columns else "Facility"
            
            fig.add_trace(go.Scattermapbox(
                lon=facility_gdf_points.geometry.x, lat=facility_gdf_points.geometry.y,
                mode='markers', marker=go.scattermapbox.Marker(
                    size=size_data, sizemin=4, color='#1E3A8A', opacity=0.9, allowoverlap=True),
                text=hover_text_data, hoverinfo='text', name='Health Facilities'
            ))
        else: logger.warning("Facility GDF provided but contains no Point geometries.")
    
    # Auto-calculate center and zoom if not provided and GDF has valid geometries
    if (center_lat is None or center_lon is None) and not gdf.empty and gdf.geometry.is_valid.all() and not gdf.geometry.is_empty.all():
        try:
            # Ensure no empty or invalid geometries before total_bounds
            valid_geoms_gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
            if not valid_geoms_gdf.empty:
                bounds = valid_geoms_gdf.total_bounds # minx, miny, maxx, maxy
                if not np.isinf(bounds).any() and not np.isnan(bounds).any(): # Check for valid bounds
                    center_lon = (bounds[0] + bounds[2]) / 2
                    center_lat = (bounds[1] + bounds[3]) / 2
                    # Very rough zoom adjustment based on longitudinal extent
                    lon_extent = bounds[2] - bounds[0]
                    if lon_extent > 0: # Ensure lon_extent is positive and not zero
                        # This is highly empirical, adjust as needed
                        # zoom_level = max(1, min(15, np.floor(8 - np.log2(lon_extent)))) # Example heuristic
                        pass # Using default zoom_level if this heuristic is too complex or unreliable
                    else: # Single point or very small extent
                        zoom_level = 12 # Zoom in closer
                else: # Fallback if bounds are not finite
                    logger.warning(f"Could not calculate valid bounds for map '{title}'. Using default center/zoom.")
            else:  logger.warning(f"No valid geometries in GDF for map '{title}' to calculate center. Using default.")
        except Exception as e_map_center:
            logger.error(f"Error calculating map center for '{title}': {e_map_center}. Using default.")

    fig.update_layout(
        title_text=title,
        mapbox_zoom=zoom_level, # Uses default from app_config if not calculated
        mapbox_center={"lat": center_lat, "lon": center_lon} if center_lat is not None and center_lon is not None else None,
        height=height, margin={"r":10,"t":50,"l":10,"b":10}, # Adjusted margins for map
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)')
    )
    logger.info(f"Rendered choropleth map: {title}")
    return fig
