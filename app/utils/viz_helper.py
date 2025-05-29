import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_annotated_line_chart(labels, data, title, color, target_line=None, target_label=None, ci_lower=None, ci_upper=None):
    """
    Create a line chart with annotations and optional confidence intervals.
    """
    if not labels or not data or len(labels) != len(data):
        st.error("Invalid input: Labels and data must be non-empty and of equal length.")
        logger.error("Invalid input for line chart")
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=labels,
        y=data,
        mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        name=title
    ))
    
    if ci_lower and ci_upper and len(ci_lower) == len(ci_upper) == len(labels):
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(150, 150, 150, 0.2)",
            line=dict(width=0),
            name="CI",
            showlegend=True
        ))
    
    if target_line is not None:
        fig.add_hline(
            y=target_line,
            line_dash="dash",
            line_color="red",
            annotation_text=target_label,
            annotation_position="top right",
            annotation_font_color="red"
        )
    
    try:
        anomalies = [i for i, v in enumerate(data) if v > np.percentile(data, 95)]
        for idx in anomalies:
            fig.add_annotation(
                x=labels[idx],
                y=data[idx],
                text="Anomaly",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30
            )
    except Exception as e:
        st.warning(f"Error detecting anomalies: {str(e)}")
        logger.warning(f"Anomaly detection error: {str(e)}")
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis=dict(title="Date", tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Value", tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=300
    )
    
    logger.info(f"Rendered line chart: {title}")
    return fig

def plot_bar_chart(categories, values, title, color):
    """
    Create a bar chart for categorical data.
    """
    if not categories or not values or len(categories) != len(values):
        st.error("Invalid input: Categories and values must be non-empty and of equal length.")
        logger.error("Invalid input for bar chart")
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=color,
        text=values,
        textposition="auto"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis=dict(title="Category", tickfont=dict(size=12)),
        yaxis=dict(title="Count", tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=300
    )
    
    logger.info(f"Rendered bar chart: {title}")
    return fig

def plot_donut_chart(labels, values, title):
    """
    Create a donut chart with hover details.
    """
    if not labels or not values or len(labels) != len(values):
        st.error("Invalid input: Labels and values must be non-empty and of equal length.")
        logger.error("Invalid input for donut chart")
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=["#d73027", "#1a9850", "#fee08b"],
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=300
    )
    
    logger.info(f"Rendered donut chart: {title}")
    return fig

def plot_heatmap(matrix, title):
    """
    Create a heatmap with annotations.
    """
    try:
        if matrix.empty:
            raise ValueError("Matrix is empty")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        if not np.all(matrix.apply(lambda x: np.isreal(x))):
            raise ValueError("Matrix contains non-numeric values")
        
        z = np.array(matrix.values, dtype=float)
        text = np.around(z, decimals=2)
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=matrix.columns,
            y=matrix.index,
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=40, t=60, b=40),
            height=300
        )
        
        logger.info(f"Rendered heatmap: {title}")
        return fig
    
    except Exception as e:
        st.error(f"Error plotting heatmap: {str(e)}")
        logger.error(f"Heatmap error: {str(e)}")
        return go.Figure()

def plot_treemap(labels, values, parents, title):
    """
    Create a treemap for prioritization.
    """
    if not labels or not values or not parents or len(labels) != len(values) or len(labels) != len(parents):
        st.error("Invalid input: Labels, values, and parents must be non-empty and of equal length.")
        logger.error("Invalid input for treemap")
        return go.Figure()
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        values=values,
        parents=parents,
        marker_colorscale="Blues",
        textinfo="label+value"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        margin=dict(l=40, r=40, t=60, b=40),
        height=300
    )
    
    logger.info(f"Rendered treemap: {title}")
    return fig

def plot_layered_choropleth_map(geojson_data, data, location_col, value_col, facility_col, title):
    """
    Create a layered choropleth map with risk and facilities, styled for realism.
    """
    if not geojson_data or data.empty or location_col not in data or value_col not in data or facility_col not in data:
        st.error("Invalid input: GeoJSON and data must be valid with required columns.")
        logger.error("Invalid input for choropleth map")
        return go.Figure()
    
    try:
        fig = px.choropleth(
            data,
            geojson=geojson_data,
            locations=location_col,
            featureidkey="properties.zone",
            color=value_col,
            color_continuous_scale="Reds",
            range_color=[2.0, 3.5],
            labels={value_col: "Disease Risk (per 1,000)"},
            hover_data=[value_col, facility_col]
        )
        
        # Add facility scatter
        facility_lons = []
        facility_lats = []
        facility_texts = []
        for feature in geojson_data["features"]:
            zone = feature["properties"]["zone"]
            zone_data = data[data[location_col] == zone]
            if not zone_data.empty:
                num_facilities = zone_data[facility_col].iloc[0]
                # Use centroid of polygon for facility marker
                coords = feature["geometry"]["coordinates"][0]  # Assuming Polygon
                lon = sum(p[0] for p in coords) / len(coords)
                lat = sum(p[1] for p in coords) / len(coords)
                facility_lons.append(lon)
                facility_lats.append(lat)
                facility_texts.append(f"{zone}: {num_facilities} facilities")
        
        fig.add_scattergeo(
            lon=facility_lons,
            lat=facility_lats,
            text=facility_texts,
            mode="markers+text",
            marker=dict(size=10, color="blue", symbol="cross"),
            textposition="top center",
            name="Health Facilities"
        )
        
        # Add basemap and styling
        fig.update_geos(
            projection_type="mercator",
            fitbounds="locations",
            visible=True,
            showcountries=True,
            showsubunits=True,
            showlakes=True,
            lakecolor="rgb(200, 200, 255)",
            showrivers=True,
            rivercolor="rgb(200, 200, 255)"
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
            margin=dict(l=20, r=20, t=60, b=20),
            height=500,
            geo=dict(
                bgcolor="white",
                landcolor="rgb(240, 240, 240)",
                subunitcolor="rgb(200, 200, 200)",
                countrycolor="rgb(150, 150, 150)"
            ),
            coloraxis_colorbar=dict(
                title="Disease Risk",
                thickness=10,
                x=0.05,
                xanchor="left",
                y=0.5,
                len=0.75
            ),
            showlegend=True
        )
        
        # Add OpenStreetMap tiles
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center={"lat": sum(facility_lats) / len(facility_lats), "lon": sum(facility_lons) / len(facility_lons)},
                "zoom=0,
                zoom="5,
                layers=[{
                    "sourcetype": "geojson",
                    "source": geojson_data,
                    "type": "fill",
                    "color": "rgba(255, 0, 0, 0.2)"
                }]
            )
        )
        
        logger.info(f"Rendered choropleth map: {title}")
        return fig
    except Exception as e:
        st.error(f"Error rendering choropleth map: {str(e)}")
        logger.error(f"Choropleth map error: {str(e)}")
        return None

def render_kpi_card(title, value, icon, status=None):
    """
    Render a styled KPI card.
    """
    status_color = {
        "High": "bg-red-500 text-white",
        "Moderate": "bg-yellow-400 text-black",
        "Low": "bg-green-500 text-white"
    }.get(status, "bg-gray-100 text-gray-800")
    
    st.markdown(
        f"""
        <div class="p-4 rounded-lg shadow-md hover:shadow-lg transition-shadow {status_color}">
            <div class="flex items-center">
                <span class="text-2xl mr-2">{icon}</span>
                <div>
                    <h3 class="text-sm font-semibold">{title}</h3>
                    <p class="text-lg font-bold">{value}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_traffic_light(message, status):
    """
    Render a traffic light indicator.
    """
    status_color = {
        "High": "bg-red-500",
        "Moderate": "bg-yellow-400",
        "Low": "bg-green-500"
    }.get(status, "bg-gray-500")
    
    st.markdown(
        f"""
        <div class="flex items-center p-2 rounded-md bg-gray-50 mb-2">
            <span class="w-4 h-4 rounded-full {status_color} mr-2"></span>
            <span class="text-sm">{message}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
