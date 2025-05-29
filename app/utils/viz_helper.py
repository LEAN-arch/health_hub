import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def plot_line_chart(labels, data, title, color, target_line=None, target_label=None):
    """
    Create a gradient-filled line chart with optional target line.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=labels,
        y=data,
        mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
        name=title
    ))
    
    if target_line is not None:
        fig.add_hline(
            y=target_line,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=target_label,
            annotation_position="top right",
            annotation_font_color="#ef4444"
        )
    
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
    
    return fig

def plot_bar_chart(categories, values, title, color):
    """
    Create a bar chart for categorical data.
    """
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
    
    return fig

def plot_donut_chart(labels, values, title):
    """
    Create a donut chart for distribution data.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=["#ef4444", "#22c55e", "#facc15"],
        textinfo="label+percent",
        hoverinfo="label+value"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=300
    )
    
    return fig

def plot_heatmap(matrix, title):
    """
    Create a heatmap for correlation matrix.
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="RdYlGn",
        zmin=-1,
        zmax=1,
        text=matrix.values.round(2),
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
    
    return fig

def plot_choropleth_map(geojson, data, location_col, value_col, title):
    """
    Create a choropleth map for geospatial data.
    """
    fig = px.choropleth(
        data,
        geojson=geojson,
        locations=location_col,
        featureidkey="properties.zone",
        color=value_col,
        color_continuous_scale="Reds",
        range_color=[2.0, 3.5],
        labels={value_col: "Disease Risk"}
    )
    
    fig.update_geos(
        fitbounds="locations",
        visible=False
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

def render_kpi_card(title, value, icon, status=None, drilldown=False):
    """
    Render a styled KPI card with optional status and drilldown.
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
                    {'<a href="#" class="text-blue-600 text-sm">Details</a>' if drilldown else ''}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_traffic_light(message, status):
    """
    Render a traffic light indicator for alerts.
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