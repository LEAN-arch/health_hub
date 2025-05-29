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

# Configure logging
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

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
def render_kpi_card(title, value, icon, status=None, delta=None, delta_type="neutral", help_text=None):
    status_class_map = {"High": "status-high", "Moderate": "status-moderate", "Low": "status-low"}
    status_final_class = status_class_map.get(status, "")
    delta_html = f'<p class="kpi-delta {delta_type}">{html.escape(str(delta))}</p>' if delta else ""
    tooltip_html = f'title="{html.escape(str(help_text))}"' if help_text else ''
    html_content = f"""<div class="kpi-card {status_final_class}" {tooltip_html}> <div class="kpi-card-header"> <span class="kpi-icon">{icon}</span> <h3 class="kpi-title">{html.escape(str(title))}</h3> </div> <div> <p class="kpi-value">{html.escape(str(value))}</p> {delta_html} </div></div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message, status, details=""):
    status_class_map = {"High": "status-high", "Moderate": "status-moderate", "Low": "status-low", "Neutral": "status-neutral"}
    dot_status_class = status_class_map.get(status, "status-neutral")
    details_html = f'<span class="traffic-light-details">{html.escape(str(details))}</span>' if details else ""
    html_content = f"""<div class="traffic-light-indicator"> <span class="traffic-light-dot {dot_status_class}"></span> <span class="traffic-light-message">{html.escape(str(message))}</span> {details_html}</div>""".strip()
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
    line_color = color if color else pio.templates[pio.templates.default].layout.colorway[0]

    fig.add_trace(go.Scatter(
        x=data_series.index, y=data_series.values, mode="lines+markers", name=y_axis_title,
        line=dict(color=line_color, width=2.5), marker=dict(size=6, symbol='circle-open'),
        hoverinfo='x+y', customdata=[y_axis_title]*len(data_series), 
        hovertemplate='<b>Date</b>: %{x}<br><b>'+y_axis_title+'</b>: %{y}<extra></extra>'
    ))

    if show_ci and lower_bound_series is not None and upper_bound_series is not None and \
       not lower_bound_series.empty and not upper_bound_series.empty:
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
            y=target_line, line_dash="dash", line_color="#EF4444",
            annotation_text=target_label if target_label else f"Target: {target_line}",
            annotation_position="bottom right", annotation_font_size=10, annotation_font_color="#EF4444"
        )

    if show_anomalies and len(data_series) > 5: 
        q95 = data_series.quantile(0.95)
        q05 = data_series.quantile(0.05)
        std_dev = data_series.std()
        mean_val = data_series.mean()
        if pd.notna(std_dev) and pd.notna(mean_val) and std_dev > 1e-6: 
            upper_anomaly_thresh = max(q95, mean_val + 2*std_dev)
            lower_anomaly_thresh = min(q05, mean_val - 2*std_dev)

            anomalies = data_series[(data_series > upper_anomaly_thresh) | (data_series < lower_anomaly_thresh)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values, mode='markers',
                    marker=dict(color='#D90429', size=10, symbol='x-thin-open', line=dict(width=2)),
                    name='Potential Anomaly', hoverinfo='x+y+text',
                    text=[f"Anomaly ({val:.2f})" for val in anomalies.values]
                ))
        else:
            logger.debug(f"Could not calculate std_dev/mean for anomaly detection in '{title}' or std_dev too small. Skipping.")

    fig.update_layout(
        title_text=title,
        xaxis_title=data_series.index.name if data_series.index.name else "Date",
        yaxis_title=y_axis_title,
        height=height,
        hovermode="x unified",
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

    if sort_values_by and sort_values_by in df_to_plot.columns:
        try:
            df_to_plot.sort_values(by=sort_values_by, ascending=ascending, inplace=True)
        except Exception as e_sort: 
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
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 20
    if data_df.empty or labels_col not in data_df.columns or values_col not in data_df.columns:
        logger.warning(f"Empty or invalid data for donut chart: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No data)", height=height, xaxis={'visible': False}, yaxis={'visible': False},
                                          annotations=[dict(text="No data to display", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    fig = go.Figure(data=[go.Pie(
        labels=data_df[labels_col], values=data_df[values_col], hole=0.45,
        pull=[0.02] * len(data_df), textinfo='label+percent', hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#ffffff', width=1.5))
    )])
    
    fig.update_layout(title_text=title, height=height, showlegend=True, 
                      legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="right", x=1.1))
    return fig


def plot_heatmap(matrix_df, title, height=None, colorscale="RdBu_r", zmid=0):
    height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 70
    
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty: 
        logger.error(f"Invalid input for heatmap: {title}. Must be a non-empty DataFrame.")
        return go.Figure().update_layout(title_text=f"{title} (No data or invalid data)", height=height,
                                          annotations=[dict(text="Invalid data for Heatmap", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
    
    try:
        is_all_numeric_convertible = True
        numeric_matrix_df_test = matrix_df.copy()
        for col in numeric_matrix_df_test.columns:
            original_non_na = matrix_df[col].dropna()
            converted_col = pd.to_numeric(matrix_df[col], errors='coerce')
            converted_non_na = converted_col.dropna()
            if len(original_non_na) != len(converted_non_na):
                is_all_numeric_convertible = False
                break
        
        if not is_all_numeric_convertible: 
             logger.error(f"Matrix for heatmap '{title}' contains non-numeric values that cannot be reliably converted.")
             return go.Figure().update_layout(title_text=f"{title} (Contains non-convertible non-numeric data)", height=height,
                                              annotations=[dict(text="Non-numeric data in Heatmap", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
        
        numeric_matrix_df = matrix_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    except Exception as e: 
        logger.error(f"Error converting matrix to numeric for heatmap '{title}': {e}", exc_info=True)
        return go.Figure().update_layout(title_text=f"{title} (Data processing error for heatmap)", height=height,
                                          annotations=[dict(text="Error processing Heatmap data", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    zmid_val = None
    if not numeric_matrix_df.empty:
        # Ensure min/max are calculated on actual numeric data if matrix_df had mixed types initially
        matrix_for_min_max = numeric_matrix_df.select_dtypes(include=np.number)
        if not matrix_for_min_max.empty:
            min_val_series = matrix_for_min_max.min()
            max_val_series = matrix_for_min_max.max()
            if not min_val_series.empty and not max_val_series.empty:
                 min_val = min_val_series.min()
                 max_val = max_val_series.max()
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


def plot_layered_choropleth_map(gdf, value_col, title,
                                 id_col='zone_id', featureidkey_prop='zone_id',
                                 color_continuous_scale="OrRd", hover_cols=None,
                                 facility_gdf=None, facility_size_col=None, facility_hover_name=None,
                                 height=None, center_lat=None, center_lon=None, zoom_level=None,
                                 mapbox_style="carto-positron"):
    height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    
    if gdf is None or gdf.empty or value_col not in gdf.columns or id_col not in gdf.columns:
        logger.warning(f"Invalid GeoDataFrame or missing columns for choropleth map: {title}")
        return go.Figure().update_layout(title_text=f"{title} (No geographic data or required metric missing)", height=height,
                                          annotations=[dict(text="Map data unavailable", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    gdf_plot = gdf.copy()
    if not pd.api.types.is_numeric_dtype(gdf_plot[value_col]):
        logger.info(f"Value column '{value_col}' for choropleth map '{title}' is not numeric. Attempting conversion.")
        gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors='coerce')
        if gdf_plot[value_col].isnull().all():
            logger.warning(f"All values in '{value_col}' became NaN after conversion for map '{title}'. Cannot plot.")
            return go.Figure().update_layout(title_text=f"{title} (No valid numeric data for selected metric)", height=height,
                                              annotations=[dict(text="Map data values are all invalid", xref="paper", yref="paper", showarrow=False, font=dict(size=14))])

    default_hover_cols = ['name', 'population', value_col]
    final_hover_cols = [col for col in (hover_cols if hover_cols else default_hover_cols) if col in gdf_plot.columns]

    # --- Mapbox Layout Arguments Construction ---
    # This dict will be passed as **kwargs to px.choropleth_mapbox
    # It will also be used to update the layout of the figure later.
    mapbox_args = {"style": mapbox_style}

    if center_lat is not None and center_lon is not None:
        mapbox_args["center"] = {"lat": center_lat, "lon": center_lon}
        mapbox_args["zoom"] = zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM
        # When center and zoom are explicitly set, we don't want 'fitbounds'
    elif zoom_level is not None: # Only zoom is given
        mapbox_args["zoom"] = zoom_level
        # Let Plotly auto-center (fitbounds will not be used here)
    else: # Neither center nor zoom provided, use fitbounds
        # No need to add 'fitbounds' directly to mapbox_args for px.choropleth_mapbox,
        # it's a direct argument to the function itself.
        pass


    # Create the figure using px.choropleth_mapbox
    # The 'fitbounds' argument to px.choropleth_mapbox handles auto-fitting.
    # It should only be used if center/zoom are NOT explicitly set.
    if "center" in mapbox_args or "zoom" in mapbox_args:
        # If we have explicit center or zoom, don't use fitbounds in the function call
        fig = px.choropleth_mapbox(
            gdf_plot, geojson=gdf_plot.geometry.__geo_interface__, locations=gdf_plot[id_col],
            featureidkey=f"properties.{featureidkey_prop}", color=value_col,
            color_continuous_scale=color_continuous_scale,
            opacity=0.75, hover_name="name" if "name" in gdf_plot.columns else id_col,
            hover_data=final_hover_cols, labels={value_col: value_col.replace('_',' ').title()},
            mapbox_style=mapbox_args.get("style"), # Use .get for safety
            center=mapbox_args.get("center"),
            zoom=mapbox_args.get("zoom")
        )
    else:
        # Use fitbounds if no explicit center/zoom
        fig = px.choropleth_mapbox(
            gdf_plot, geojson=gdf_plot.geometry.__geo_interface__, locations=gdf_plot[id_col],
            featureidkey=f"properties.{featureidkey_prop}", color=value_col,
            color_continuous_scale=color_continuous_scale,
            opacity=0.75, hover_name="name" if "name" in gdf_plot.columns else id_col,
            hover_data=final_hover_cols, labels={value_col: value_col.replace('_',' ').title()},
            mapbox_style=mapbox_args.get("style"),
            fitbounds="locations" # Main change for auto-fitting
        )


    # Add facility markers if provided
    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        facility_gdf_points = facility_gdf[facility_gdf.geometry.geom_type == 'Point'].copy()
        if not facility_gdf_points.empty:
            size_data = 8
            if facility_size_col and facility_size_col in facility_gdf_points.columns and pd.api.types.is_numeric_dtype(facility_gdf_points[facility_size_col]):
                min_s, max_s = 4, 15
                s_min_val, s_max_val = facility_gdf_points[facility_size_col].min(), facility_gdf_points[facility_size_col].max()
                if pd.notna(s_min_val) and pd.notna(s_max_val) and s_max_val > s_min_val :
                    size_data = min_s + (facility_gdf_points[facility_size_col] - s_min_val) * (max_s - min_s) / (s_max_val - s_min_val)
                elif pd.notna(s_max_val) and s_max_val == s_min_val:
                    size_data = (min_s + max_s) / 2
            
            hover_text_data = facility_gdf_points[facility_hover_name] if facility_hover_name and facility_hover_name in facility_gdf_points.columns else "Facility"
            
            fig.add_trace(go.Scattermapbox(
                lon=facility_gdf_points.geometry.x, lat=facility_gdf_points.geometry.y,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size_data, sizemin=4, color='#1E3A8A', opacity=0.85, allowoverlap=True),
                text=hover_text_data, hoverinfo='text', name='Health Facilities'
            ))
        else: logger.warning("Facility GDF provided but contains no Point geometries for map layer.")

    # Final layout updates
    fig.update_layout(
        title_text=title,
        height=height,
        margin={"r":5,"t":45,"l":5,"b":5}, # Consistent tight margins
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.6)')
    )
    # The mapbox_style, center, and zoom are now primarily handled by the px.choropleth_mapbox call arguments
    # If you needed to update them *after* creation, you would use:
    # fig.update_layout(mapbox=mapbox_args)
    # However, it's better to pass them directly if possible.

    logger.info(f"Rendered choropleth map: {title}")
    return fig
    
    fig.update_layout(title_text=title, height=height, xaxis_showgrid=False, yaxis_showgrid=False, 
                      xaxis_tickangle=-30 if len(numeric_matrix_df.columns) > 5 else 0, yaxis_autorange='reversed')
    # logger.info(f"Rendered heatmap: {title}") # Reduce noise
    return fig
