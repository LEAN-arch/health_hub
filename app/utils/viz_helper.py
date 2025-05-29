import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Plotly Theme ---
def set_custom_plotly_theme():
    custom_theme = go.layout.Template()
    # Font and Colors
    custom_theme.layout.font = dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size=12, color="#374151") # Gray-700
    custom_theme.layout.paper_bgcolor = "#FFFFFF"
    custom_theme.layout.plot_bgcolor = "#FFFFFF"
    custom_theme.layout.colorway = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#6366F1', '#8B5CF6'] # Blue, Green, Amber, Red, Indigo, Purple

    # Axes
    custom_theme.layout.xaxis = dict(
        gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#E5E7EB", # Gray-200, Gray-300
        title_font_size=13, tickfont_size=11, automargin=True
    )
    custom_theme.layout.yaxis = dict(
        gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#E5E7EB",
        title_font_size=13, tickfont_size=11, automargin=True
    )
    # Title (left-aligned by default in Plotly now if x not specified, or use x=0.01 for slight padding)
    custom_theme.layout.title = dict(font_size=16, font_weight='bold', x=0.02, xanchor='left', y=0.95, yanchor='top')
    
    # Legend
    custom_theme.layout.legend = dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#E5E7EB', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)

    # Margins (can be overridden per plot)
    custom_theme.layout.margin = dict(l=50, r=30, t=60, b=50) # Adjusted for titles and axes

    pio.templates["custom_health_theme"] = custom_theme
    pio.templates.default = "plotly+custom_health_theme"
    logger.info("Custom Plotly theme 'custom_health_theme' set as default.")

set_custom_plotly_theme() # Apply theme when module is imported

# --- Styled Components (HTML/CSS via st.markdown) ---

def render_kpi_card(title, value, icon, status=None, delta=None, delta_type="neutral", help_text=None):
    """
    Render a styled KPI card using CSS classes defined in style.css.
    delta: string like "+5%" or "-2 days"
    delta_type: "positive", "negative", "neutral"
    """
    status_class = {
        "High": "status-high", "Moderate": "status-moderate", "Low": "status-low"
    }.get(status, "")

    delta_html = ""
    if delta:
        delta_html = f'<p class="kpi-delta {delta_type}">{delta}</p>'

    tooltip_html = f'title="{help_text}"' if help_text else ''

    html_content = f"""
    <div class="kpi-card {status_class}" {tooltip_html}>
        <div class="kpi-card-header">
            <span class="kpi-icon">{icon}</span>
            <h3 class="kpi-title">{title}</h3>
        </div>
        <div>
            <p class="kpi-value">{value}</p>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message, status, details=""):
    """
    Render a traffic light indicator using CSS classes.
    status: "High", "Moderate", "Low", "Neutral"
    """
    status_class = f"status-{status.lower()}" if status else "status-neutral"
    details_html = f'<span class="traffic-light-details">{details}</span>' if details else ""

    html_content = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {status_class}"></span>
        <span class="traffic-light-message">{message}</span>
        {details_html}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

# --- Plotting Functions ---

def plot_annotated_line_chart(data_series, title, y_axis_title="Value", color=None,
                              target_line=None, target_label=None, show_ci=False, 
                              lower_bound_series=None, upper_bound_series=None,
                              height=350):
    """
    Create an enhanced line chart with optional annotations, CI.
    data_series: Pandas Series with DatetimeIndex.
    """
    if not isinstance(data_series, pd.Series) or data_series.empty:
        # st.warning(f"No data for line chart: {title}")
        logger.warning(f"Empty data_series for line chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height)

    fig = go.Figure()
    line_color = color if color else pio.templates.default.layout.colorway[0]

    # Main line
    fig.add_trace(go.Scatter(
        x=data_series.index,
        y=data_series.values,
        mode="lines+markers",
        name=y_axis_title, # Or data_series.name if it exists
        line=dict(color=line_color, width=2.5),
        marker=dict(size=6, symbol='circle'),
        hoverinfo='x+y'
    ))

    # Confidence Interval
    if show_ci and lower_bound_series is not None and upper_bound_series is not None and \
       not lower_bound_series.empty and not upper_bound_series.empty:
        fig.add_trace(go.Scatter(
            x=list(data_series.index) + list(data_series.index[::-1]), # x, then x reversed
            y=list(upper_bound_series.values) + list(lower_bound_series.values[::-1]), # upper, then lower reversed
            fill="toself",
            fillcolor=f"rgba({int(line_color[1:3],16)},{int(line_color[3:5],16)},{int(line_color[5:7],16)},0.15)", # Lighter shade of line_color
            line=dict(width=0),
            name="Confidence Interval",
            hoverinfo='skip'
        ))
    
    # Target Line
    if target_line is not None:
        fig.add_hline(
            y=target_line,
            line_dash="dash",
            line_color="#EF4444", # Red-500 for critical targets
            annotation_text=target_label if target_label else f"Target: {target_line}",
            annotation_position="bottom right",
            annotation_font_size=10,
            annotation_font_color="#EF4444"
        )
    
    # Anomalies (Simple example: points > 95th percentile or < 5th)
    # In a real app, anomalies would come from a dedicated detection algorithm
    q95 = data_series.quantile(0.95)
    q05 = data_series.quantile(0.05)
    anomalies = data_series[(data_series > q95) | (data_series < q05)]
    
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies.values,
            mode='markers',
            marker=dict(color='#EF4444', size=10, symbol='x'),
            name='Potential Anomaly',
            hoverinfo='x+y+text',
            text=['Anomaly'] * len(anomalies)
        ))

    fig.update_layout(
        title_text=title,
        xaxis_title=data_series.index.name if data_series.index.name else "Date",
        yaxis_title=y_axis_title,
        height=height,
        hovermode="x unified", # Improved hover experience
        legend=dict(traceorder='normal')
    )
    logger.info(f"Rendered line chart: {title}")
    return fig


def plot_bar_chart(df, x_col, y_col, title, color_col=None, barmode='group',
                   orientation='v', y_axis_title=None, x_axis_title=None, height=350):
    """
    Create a versatile bar chart. df is a pandas DataFrame.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Empty or invalid data for bar chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height)

    y_axis_title = y_axis_title if y_axis_title else y_col.replace('_', ' ').title()
    x_axis_title = x_axis_title if x_axis_title else x_col.replace('_', ' ').title()

    fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col,
                 barmode=barmode, orientation=orientation, height=height,
                 labels={y_col: y_axis_title, x_col: x_axis_title})
    
    fig.update_traces(marker_line_width=0.8, marker_line_color='rgba(0,0,0,0.5)',
                      texttemplate='%{y}', textposition='outside') # Show values outside bars
    fig.update_layout(yaxis_title=y_axis_title, xaxis_title=x_axis_title)
    
    logger.info(f"Rendered bar chart: {title}")
    return fig


def plot_donut_chart(data, labels_col, values_col, title, height=380):
    """
    Create a donut chart. data is a pandas DataFrame.
    """
    if data.empty or labels_col not in data.columns or values_col not in data.columns:
        logger.warning(f"Empty or invalid data for donut chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height)

    fig = go.Figure(data=[go.Pie(
        labels=data[labels_col],
        values=data[values_col],
        hole=0.45, # Increased hole size for more "donut" look
        pull=[0.02] * len(data), # Slight pull for segments
        textinfo='label+percent', # label+percent+value
        hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#ffffff', width=1.5)) # White lines between segments
    )])
    
    fig.update_layout(title_text=title, height=height, showlegend=True) # Legend can be useful for many categories
    logger.info(f"Rendered donut chart: {title}")
    return fig

def plot_heatmap(matrix_df, title, height=400, colorscale="RdYlGn_r"): # Reversed RedYellowGreen
    """
    Create a heatmap from a pandas DataFrame (correlation matrix or similar).
    """
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        logger.error(f"Invalid input for heatmap: {title}. Must be a non-empty DataFrame.")
        # st.error(f"Cannot plot heatmap '{title}': Data is empty or not a DataFrame.")
        return go.Figure().update_layout(title_text=f"{title} (No data or invalid data)", height=height)
    
    if matrix_df.shape[0] != matrix_df.shape[1] and title.lower().endswith("correlations"): # Heuristic
         logger.warning(f"Matrix for heatmap '{title}' is not square ({matrix_df.shape}). This might be an issue for correlation heatmaps.")
         # Not raising error, could be an intentional non-square heatmap
    
    # Ensure all data is numeric, try to convert if not, else error
    try:
        numeric_matrix_df = matrix_df.apply(pd.to_numeric, errors='coerce')
        if numeric_matrix_df.isnull().values.any(): # Coercion resulted in NaNs
            logger.error(f"Matrix for heatmap '{title}' contains non-numeric values that could not be converted.")
            # st.error(f"Cannot plot heatmap '{title}': Contains non-numeric data.")
            return go.Figure().update_layout(title_text=f"{title} (Contains non-numeric data)", height=height)
    except Exception as e:
        logger.error(f"Error converting matrix to numeric for heatmap '{title}': {e}")
        # st.error(f"Error processing data for heatmap '{title}'.")
        return go.Figure().update_layout(title_text=f"{title} (Data processing error)", height=height)

    fig = go.Figure(data=go.Heatmap(
        z=numeric_matrix_df.values,
        x=numeric_matrix_df.columns.tolist(),
        y=numeric_matrix_df.index.tolist(),
        colorscale=colorscale,
        zmin=-1 if title.lower().endswith("correlations") else None, # For correlations
        zmax=1 if title.lower().endswith("correlations") else None,
        text=np.around(numeric_matrix_df.values, decimals=2),
        texttemplate="%{text}",
        hoverongaps=False,
        xgap=1, ygap=1 # Small gaps between cells
    ))
    
    fig.update_layout(
        title_text=title,
        height=height,
        xaxis_showgrid=False, yaxis_showgrid=False, # Cleaner look
        xaxis_tickangle=-30 # Angle ticks if long
    )
    logger.info(f"Rendered heatmap: {title}")
    return fig


def plot_layered_choropleth_map(gdf, value_col, title, 
                                 id_col='zone_id', # The column in gdf used for 'locations'
                                 featureidkey_prop='zone_id', # The property in GeoJSON's 'properties' dict
                                 color_continuous_scale="OrRd", # Oranges to Reds
                                 hover_cols=None, facility_gdf=None, 
                                 facility_size_col=None, facility_hover_name=None,
                                 height=600, center_lat=None, center_lon=None, zoom_level=8):
    """
    Create a choropleth map, optionally with facility markers.
    gdf: GeoDataFrame with geometries and data.
    value_col: Column in gdf to use for choropleth colors.
    facility_gdf: Optional GeoDataFrame for facility markers (must have 'geometry' with Points).
    """
    if gdf is None or gdf.empty or value_col not in gdf.columns or id_col not in gdf.columns:
        logger.warning(f"Invalid GeoDataFrame or missing columns for choropleth map: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data or invalid GeoDataFrame)", height=height)

    # Use choropleth_mapbox for base map tiles
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry.__geo_interface__, # Provide geometry directly
        locations=gdf[id_col],                  # Column in gdf to link features
        featureidkey=f"properties.{featureidkey_prop}", # Path to ID in GeoJSON properties
        color=value_col,
        color_continuous_scale=color_continuous_scale,
        mapbox_style="carto-positron", # "open-street-map", "carto-positron", "stamen-terrain"
        opacity=0.7,
        hover_name="name" if "name" in gdf.columns else id_col, # Display zone name on hover
        hover_data=hover_cols if hover_cols else [value_col] # Additional data on hover
    )

    # Add facility markers if provided
    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        if not facility_gdf.geometry.geom_type.isin(['Point']).all():
            logger.warning("Facility GDF contains non-Point geometries. Skipping facility layer.")
        else:
            fig.add_trace(go.Scattermapbox(
                lon=facility_gdf.geometry.x,
                lat=facility_gdf.geometry.y,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=facility_gdf[facility_size_col] if facility_size_col and facility_size_col in facility_gdf.columns else 8,
                    color='#1E3A8A', # Dark Blue
                    opacity=0.8
                ),
                text=facility_gdf[facility_hover_name] if facility_hover_name and facility_hover_name in facility_gdf.columns else "Facility",
                hoverinfo='text',
                name='Health Facilities'
            ))
    
    # Calculate center and zoom if not provided
    if center_lat is None or center_lon is None:
        if not gdf.empty:
            # Get bounds of all geometries to calculate center
            bounds = gdf.total_bounds # minx, miny, maxx, maxy
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            # Basic zoom adjustment (very rough)
            # zoom_level = np.log2(360 / (bounds[2] - bounds[0])) if (bounds[2] - bounds[0]) > 0 else 8
        else: # Default center if gdf is empty but we somehow got here
            center_lat, center_lon = 0, 0


    fig.update_layout(
        title_text=title,
        mapbox_zoom=zoom_level,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        height=height,
        margin={"r":0,"t":40,"l":0,"b":0}, # Tight margin for map
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    logger.info(f"Rendered choropleth map: {title}")
    return fig
