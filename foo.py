import plotly.graph_objects as go

# Sample data for the flame chart
data = [
    {"function": "A", "depth": 1, "start": 0, "duration": 5},
    {"function": "B", "depth": 2, "start": 1, "duration": 3},
    {"function": "C", "depth": 2, "start": 2, "duration": 2},
    {"function": "D", "depth": 3, "start": 1, "duration": 1},
    {"function": "E", "depth": 1, "start": 5, "duration": 2},
    {"function": "F", "depth": 2, "start": 6, "duration": 1},
]

# Create an empty figure
fig = go.Figure()

# Add rectangles for each function in the flame chart
for entry in data:
    fig.add_shape(
        type="rect",
        x0=entry["start"],
        x1=entry["start"] + entry["duration"],
        y0=entry["depth"] - 0.4,
        y1=entry["depth"] + 0.4,
        yref="y",
        xref="x",
        fillcolor="rgba(50, 171, 96, 0.8)",
        line={"width": 1},
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=[entry["start"] + entry["duration"] / 2],
            y=[entry["depth"]],
            text=[entry["function"]],
            mode="text",
            showlegend=False,
            textfont=dict(color="white", size=12),
        )
    )

# Configure the layout of the flame chart
fig.update_layout(
    title="Flame Chart",
    xaxis_title="Time",
    yaxis_title="Call Stack Depth",
    yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
    xaxis=dict(showgrid=False, zeroline=False),
    plot_bgcolor="rgba(0,0,0,0)",
)

# Show the flame chart
fig.show()
