import streamlit as st
from pulp import *
import gridstatusio
from gridstatusio import GridStatusClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.title("Battery Scheduling App")

# Define Battery Parameters

energy_capacity = st.number_input("Energy capacity (MWh)", value=100.0)
charge_power_limit = st.number_input("Charge power limit (MW)", value=25.0)
discharge_power_limit = st.number_input("Discharge power limit (MW)", value=25.0)
charge_efficiency = st.number_input("Charge efficiency", value=0.95)
discharge_efficiency = st.number_input("Discharge efficiency", value=0.95)
SOC_max = st.number_input("Max SOC (MWh)", value=100.0)
SOC_min = st.number_input("Min SOC (MWh)", value=0.0)
daily_cycle_limit = st.number_input("Daily cycle limit", value=1.0)
annual_cycle_limit = st.number_input("Annual cycle limit", value=300.0)
SOC_initial = st.number_input("Initial SOC (MWh)", value=0.0)

# Get data from gridstatus.io

API_Key = st.text_input("API Key", value="ebb576413c2308080c81d9ded9ae8c86")
client = GridStatusClient(API_Key)

pricing_node = st.text_input("Pricing Node", value="TH_NP15_GEN-APND") 
start_date = st.date_input("Start Date", value=pd.to_datetime('2022-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2023-01-01'))

grid_status_data = client.get_dataset(
    dataset="caiso_lmp_day_ahead_hourly",
    filter_column="location",
    filter_value= pricing_node,
    start = start_date.strftime('%Y-%m-%d'),
    end = end_date.strftime('%Y-%m-%d'),
    tz="US/Pacific",  # return time stamps in Pacific time
)

# Create dataframe for relevant columns and extract prices as a list from it

da_prices_df = grid_status_data[["interval_start_local", "lmp"]]
da_prices = da_prices_df['lmp'].tolist()

# Price Forecast for num_hours hours
num_hours = len(da_prices)
num_days = num_hours / 24
total_cycle_limit = (num_days / 365) * annual_cycle_limit

# Create a function to define the optimization model
def optimization_model(num_hours, da_prices):
    # Variables
    charge_vars = LpVariable.dicts("Charging", range(num_hours), lowBound=0, upBound=charge_power_limit)
    discharge_vars = LpVariable.dicts("Discharging", range(num_hours), lowBound=0, upBound=discharge_power_limit)
    SOC_vars = LpVariable.dicts("SOC", range(num_hours+1), lowBound=SOC_min, upBound=SOC_max)  # Including initial SOC

    # Problem
    prob = LpProblem("Battery Scheduling", LpMaximize)

    # Objective function
    prob += lpSum([da_prices[t]*discharge_efficiency*discharge_vars[t] - da_prices[t]*charge_vars[t]/charge_efficiency for t in range(num_hours)])

    # Constraints
    # Initial SOC constraint
    prob += SOC_vars[0] == SOC_initial

    # SOC update constraints
    for t in range(num_hours):
        if t == 0:
            prob += SOC_vars[t+1] == SOC_vars[t] + charge_vars[t] - discharge_vars[t]
        else:
            prob += SOC_vars[t+1] == SOC_vars[t] + charge_efficiency*charge_vars[t] - discharge_vars[t]
            prob += discharge_vars[t] <= discharge_efficiency * SOC_vars[t]

    # Cycle limit constraints
    prob += lpSum([charge_vars[t] for t in range(num_hours)]) <= total_cycle_limit*energy_capacity

    # Write the problem formulation for debugging
    st.write("Problem formulation:")
    st.text(str(prob))

    # Solve the problem
    prob.solve()

    # Write the status of the problem solution
    st.write("Schedule Status: {}".format(LpStatus[prob.status]))

    # Prepare data for results table
    results = []
    for t in range(num_hours):
        results.append([da_prices_df['interval_start_local'][t], charge_vars[t].varValue, discharge_vars[t].varValue, SOC_vars[t].varValue])

    results_df = pd.DataFrame(results, columns=["Time", "Charging (MW)", "Discharging (MW)", "SOC (MWh)"])
    st.dataframe(results_df)


    # Prepare data for the plots
    SOC = [x.varValue for x in list(SOC_vars.values())[:-1]]  # Exclude last SOC

    # Create subplots: SOC and Prices
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Charging Power", "Discharging Power", "State of Charge", "Day Ahead Prices"))

    fig.add_trace(go.Scatter(x=da_prices_df['interval_start_local'], y=charging, mode='lines', name='Charging'), row=1, col=1)
    fig.add_trace(go.Scatter(x=da_prices_df['interval_start_local'], y=discharging, mode='lines', name='Discharging'), row=2, col=1)
    fig.add_trace(go.Scatter(x=da_prices_df['interval_start_local'], y=SOC, mode='lines', name='SOC'), row=3, col=1)
    fig.add_trace(go.Scatter(x=da_prices_df['interval_start_local'], y=da_prices, mode='lines', name='DA Prices'), row=4, col=1)

    st.plotly_chart(fig)
