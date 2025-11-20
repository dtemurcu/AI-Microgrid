import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(page_title="Ontario Grid AI Commander", layout="wide")

try:
    df = pd.read_csv("simulation_results.csv")
    with open('config.json') as f: config = json.load(f)
    ici_manager = ICIManager(config['optimization']['historical_peak_mw'])
except Exception as e:
    st.error(f"Error loading data or config. Run 'python run_simulation.py' first.")
    st.stop()

# --- KPIs ---
st.title("⚡ AI-Enhanced Microgrid Manager (Ontario Focus)")
st.markdown("### Model Predictive Control (MPC) and Grid Service Co-Optimization")

# Calculate Financials & Operational Metrics
peak_reduction_kw = config['current_specs']['battery_max_discharge_kw']
ici_events = df['is_ici_event'].sum()
ici_savings = ici_manager.calculate_potential_savings(peak_reduction_kw)

col1, col2, col3, col4 = st.columns(4)
col1.metric("GA Avoidance Potential", f"${ici_savings:,.0f}", "Annual Class A Savings/MW")
col2.metric("Total Hours Optimized", f"{len(df)} hrs")
col3.metric("ICI Peak Events Triggered", f"{ici_events} events", delta_color="inverse")
col4.metric("Max Reactive Power", f"{df['Reactive_kVAR'].abs().max():,.0f} kVAR", "Voltage Support Capability")


# --- Main Plot (P & Q) ---
st.subheader("🔋 Power Dispatch and Grid Services")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("Active Power Dispatch (P) & Price (Cents/kWh)", "Reactive Power (Q) kVAR"))

# Row 1, Axis 1: Active Power Stack
fig.add_trace(go.Scatter(x=df.index, y=df['Grid_Import_kW'].clip(lower=0), name="Grid Import", stackgroup='one', fillcolor='rgba(100, 149, 237, 0.7)', line=dict(width=0)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Solar_kW'], name="Solar", stackgroup='one', fillcolor='rgba(255, 200, 0, 0.7)', line=dict(width=0), mode='lines'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Batt_Discharge_kW'], name="Battery Discharge", stackgroup='one', fillcolor='rgba(50, 205, 50, 0.7)', line=dict(width=0)), row=1, col=1)

# Row 1, Axis 1: Load Demand Line
fig.add_trace(go.Scatter(x=df.index, y=df['Load_kW'], name="Load Demand", line=dict(color='black', width=3, dash='dot')), row=1, col=1)

# Row 1, Axis 2: Price Line
fig.add_trace(go.Scatter(x=df.index, y=df['Price_cents'], name="Grid Price", line=dict(color='red', width=2), yaxis='y2'), row=1, col=1)

# Highlight ICI Events
ici_df = df[df['is_ici_event']]
fig.add_trace(go.Scatter(x=ici_df.index, y=ici_df['Load_kW'], 
                         mode='markers', name="ICI Trigger", 
                         marker=dict(color='red', size=10, symbol='star', line=dict(width=1, color='black'))), row=1, col=1)

# Row 2: Reactive Power (Q) & SOC
# Axis 1: Reactive Power
fig.add_trace(go.Bar(x=df.index, y=df['Reactive_kVAR'], name="Reactive kVAR", marker_color='purple'), row=2, col=1)

# Axis 2: SOC
fig.add_trace(go.Scatter(x=df.index, y=df['Batt_SOC_kWh'], name="Battery SOC", line=dict(color='darkgreen', width=3), yaxis='y3'), row=2, col=1)

# Update Layout for secondary Y-axes
fig.update_layout(height=800, hovermode="x unified", legend_tracegroupgap=250)
fig.update_yaxes(title_text="Active Power (kW)", row=1, col=1)
fig.update_yaxes(title_text="Price (cents/kWh)", secondary_y=True, row=1, col=1, anchor='x')
fig.update_yaxes(title_text="Reactive kVAR", row=2, col=1)
fig.update_yaxes(title_text="SOC (kWh)", secondary_y=True, row=2, col=1, anchor='x') # Need to map y-axes properly in Plotly

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 View Hourly Dispatch Data"):
    st.dataframe(df)
