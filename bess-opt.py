import streamlit as st
from pulp import *
import gridstatusio
from gridstatusio import GridStatusClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

#Streamlit Setup

#width
st.set_page_config(layout="wide")

#hide menu bars
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Define Pricing Info

st.header("Energy Market Inputs")

col1, col2 = st.columns(2)

pricing_nodes = ["TH_ZP26_GEN-APND", "TH_SP15_GEN-APND", "TH_NP15_GEN-APND", "DLAP_VEA-APND", "DLAP_SDGE-APND", "DLAP_SCE-APND", "DLAP_PGE-APND"]
pricing_node = st.selectbox("Pricing Node", options=pricing_nodes, index=2, key="pricing_node")

with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime('2022-01-01'), key="start_date")
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime('2023-01-01'), key="end_date")

# Define Battery Parameters

st.header("Battery Parameters")

col1, col2 = st.columns(2)

with col1:
    energy_capacity = st.slider("Energy capacity (MWh)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0, format="%.1f", key="energy_capacity")
    charge_power_limit = st.slider("Charge power limit (MW)", min_value=0.0, max_value=energy_capacity, value=25.0, step=1.0, format="%.1f", key="charge_power_limit")
    discharge_power_limit = st.slider("Discharge power limit (MW)", min_value=0.0, max_value=energy_capacity, value=25.0, step=1.0, format="%.1f", key="discharge_power_limit")
    SOC_initial = st.slider("Initial SOC (MWh)", min_value=0.0, max_value=energy_capacity, value=0.0, step=1.0, format="%.1f", key="SOC_initial")
    daily_cycle_limit = st.slider("Daily cycle limit", min_value=0.0, max_value=10.0, value=1.0, step=1.0, format="%.1f", key="daily_cycle_limit")

with col2:
    discharge_efficiency = st.slider("Discharge efficiency", min_value=0.0, max_value=1.0, value=0.95, step=0.01, format="%.2f", key="discharge_efficiency")
    charge_efficiency = st.slider("Charge efficiency", min_value=0.0, max_value=1.0, value=0.95, step=0.01, format="%.2f", key="charge_efficiency")
    SOC_max = st.slider("Max SOC (MWh)", min_value=0.0, max_value=energy_capacity, value=100.0, step=1.0, format="%.1f", key="SOC_max")
    SOC_min = st.slider("Min SOC (MWh)", min_value=0.0, max_value=energy_capacity, value=0.0, step=1.0, format="%.1f", key="SOC_min")
    annual_cycle_limit = st.slider("Annual cycle limit", min_value=0.0, max_value=daily_cycle_limit * 365, value=300.0, step=1.0, format="%.1f", key="annual_cycle_limit")

# Button to run the optimization
if st.button('Run Optimization'):

    # Make it rain
    from streamlit_extras.let_it_rain import rain
    rain(
        emoji="🔋",
        font_size=54,
        falling_speed=3,
        animation_length=1,
    )
    
    # Get data from gridstatus.io
    API_Key = "ebb576413c2308080c81d9ded9ae8c86"
    client = GridStatusClient(API_Key)

    grid_status_data = client.get_dataset(
        dataset="caiso_lmp_day_ahead_hourly",
        filter_column="location",
        filter_value=pricing_node,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
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
        SOC_vars = LpVariable.dicts("SOC", range(num_hours + 1), lowBound=SOC_min, upBound=SOC_max)  # Including initial SOC

        # Problem
        prob = LpProblem("Battery Scheduling", LpMaximize)

        # Objective function
        prob += lpSum([da_prices[t] * discharge_efficiency * discharge_vars[t] - da_prices[t] * charge_vars[t] / charge_efficiency for t in range(num_hours)])

        # Constraints
        # Initial SOC constraint
        prob += SOC_vars[0] == SOC_initial

        # SOC update constraints
        for t in range(num_hours):
            if t == 0:
                prob += SOC_vars[t + 1] == SOC_vars[t] + charge_vars[t] - discharge_vars[t]
            else:
                prob += SOC_vars[t + 1] == SOC_vars[t] + charge_efficiency * charge_vars[t] - discharge_vars[t]
                prob += discharge_vars[t] <= discharge_efficiency * SOC_vars[t]

        # Cycle limit constraints
        prob += lpSum([charge_vars[t] for t in range(num_hours)]) <= total_cycle_limit * energy_capacity / charge_efficiency

        # Solve the problem
        prob.solve()

        # Return the optimization model and variables
        return prob, charge_vars, discharge_vars, SOC_vars

    # Run the optimization model
    prob, charge_vars, discharge_vars, SOC_vars = optimization_model(num_hours, da_prices)

    # Prepare data for results table
    results = []
    for t in range(num_hours):
        results.append([da_prices_df['interval_start_local'][t], da_prices_df['lmp'][t], charge_vars[t].varValue, discharge_vars[t].varValue, SOC_vars[t].varValue])

    results_df = pd.DataFrame(results, columns=["Time", "LMP $/MWh", "Charging (MW)", "Discharging (MW)", "SOC (MWh)"])

    # Calculate hourly metrics
    results_df['Discharging Revenue ($)'] = results_df['Discharging (MW)'] * results_df["LMP $/MWh"] * discharge_efficiency
    results_df['Charging Costs ($)'] = results_df['Charging (MW)'] * results_df["LMP $/MWh"] / charge_efficiency
    results_df['Net Revenue ($)'] = results_df['Discharging Revenue ($)'] - results_df['Charging Costs ($)']
    results_df['Cycles'] = results_df['Charging (MW)'] * charge_efficiency / energy_capacity

    # Aggregate data daily, weekly, monthly, and yearly
    cols_to_keep = ['Discharging Revenue ($)', 'Charging Costs ($)', 'Net Revenue ($)', 'Cycles']

    daily_metrics = results_df.set_index('Time')[cols_to_keep].resample('D').sum()
    weekly_metrics = results_df.set_index('Time')[cols_to_keep].resample('W').sum()
    weekly_metrics.index = weekly_metrics.index - pd.DateOffset(days=6)
    monthly_metrics = results_df.set_index('Time')[cols_to_keep].resample('MS').sum()
    yearly_metrics = results_df.set_index('Time')[cols_to_keep].resample('Y').sum()
    yearly_metrics.index = yearly_metrics.index.to_period('Y').to_timestamp('Y')

    # Add start and end dates for each period
    daily_metrics['End Date'] = (daily_metrics.index + pd.DateOffset(days=1)) - pd.Timedelta(1, unit='s')
    weekly_metrics['End Date'] = (weekly_metrics.index + pd.DateOffset(weeks=1)) - pd.Timedelta(1, unit='s')
    monthly_metrics['End Date'] = (monthly_metrics.index + pd.offsets.MonthBegin(1)) - pd.Timedelta(1, unit='D')
    yearly_metrics['End Date'] = (yearly_metrics.index + pd.DateOffset(years=1)) - pd.Timedelta(1, unit='s')

    # Add start and end dates for each period
    daily_metrics['Start Date'] = daily_metrics.index
    weekly_metrics['Start Date'] = weekly_metrics.index
    monthly_metrics['Start Date'] = monthly_metrics.index
    yearly_metrics['Start Date'] = yearly_metrics.index

    # Determine table metrics based on the number of days of analysis
    if num_days <= 31:
        # Daily metrics
        metrics = daily_metrics
    elif num_days <= 93:
        # Weekly metrics
        metrics = weekly_metrics
    elif num_days <= 730:
        # Monthly metrics
        metrics = monthly_metrics
    else:
        # Yearly metrics
        metrics = yearly_metrics

    # Calculate the total for each column
    totals = metrics.sum(numeric_only=True)
    totals.name = 'Total'

    # Append totals to the end of the dataframe
    metrics = pd.concat([metrics, pd.DataFrame(totals).T])

    # Rename columns for the final table
    metrics = metrics.rename(columns={
        'Discharging Revenue ($)': 'Discharging Revenue ($)',
        'Charging Costs ($)': 'Charging Costs ($)',
        'Net Revenue ($)': 'Net Revenue ($)',
        'Cycles': 'Cycles'
    })

    # Prepare the values in pandas before passing to Plotly
    metrics_no_total = metrics.iloc[:-1].copy()  # Exclude the 'Total' row temporarily
    metrics_no_total.index = pd.to_datetime(metrics_no_total.index).strftime('%Y-%m-%d %H:%M:%S')
    metrics_no_total['End Date'] = metrics_no_total['End Date'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else '')
    metrics_no_total['Cycles'] = metrics_no_total['Cycles'].round(1)
    metrics_no_total['Discharging Revenue ($)'] = metrics_no_total['Discharging Revenue ($)'].apply(
        lambda x: f"${x:,.0f}")
    metrics_no_total['Charging Costs ($)'] = metrics_no_total['Charging Costs ($)'].apply(lambda x: f"${x:,.0f}")
    metrics_no_total['Net Revenue ($)'] = metrics_no_total['Net Revenue ($)'].apply(lambda x: f"${x:,.0f}")

    # Handle the 'Total' row separately
    total_row = metrics.iloc[-1].copy()
    total_row.name = 'Total'
    total_row['Start Date'] = ''
    total_row['End Date'] = ''
    total_row['Cycles'] = f"{total_row['Cycles']:.1f}"
    total_row['Discharging Revenue ($)'] = f"${total_row['Discharging Revenue ($)']:,.0f}"
    total_row['Charging Costs ($)'] = f"${total_row['Charging Costs ($)']:,.0f}"
    total_row['Net Revenue ($)'] = f"${total_row['Net Revenue ($)']:,.0f}"

    # Join them back together
    metrics = pd.concat([metrics_no_total, total_row.to_frame().T])

    # Reorder columns in the metrics DataFrame
    metrics = metrics.reindex(columns=['Start Date', 'End Date', 'Cycles', 'Discharging Revenue ($)',
                                       'Charging Costs ($)', 'Net Revenue ($)'])

    # Reset the index of metrics DataFrame
    metrics.reset_index(drop=True, inplace=True)

    # Calculate total metrics
    total_discharging_revenue = pd.to_numeric(metrics.loc[metrics['Start Date'] == '', 'Discharging Revenue ($)'].str.replace(',', '').str.replace('$', '')).values[0]
    total_charging_costs = pd.to_numeric(metrics.loc[metrics['Start Date'] == '', 'Charging Costs ($)'].str.replace(',', '').str.replace('$', '')).values[0]
    total_net_revenue = pd.to_numeric(metrics.loc[metrics['Start Date'] == '', 'Net Revenue ($)'].str.replace(',', '').str.replace('$', '')).values[0]
    
    # Calculate total cycles
    total_cycles = pd.to_numeric(metrics.loc[metrics['Start Date'] == '', 'Cycles']).values[0]
    
    # Calculate average net revenue per cycle
    if total_cycles != 0:
        average_profit_per_mwh = total_net_revenue / total_cycles / (energy_capacity / charge_efficiency)
    else:
        average_net_revenue_per_cycle = 0

    # Display metrics
    st.header("Optimization Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Discharging Revenue", f"${total_discharging_revenue:,.0f}")
    col2.metric("Total Charging Costs", f"${total_charging_costs:,.0f}")
    col3.metric("Total Net Revenue", f"${total_net_revenue:,.0f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Days Analyzed", f"{num_days:,.0f}")
    col5.metric("Total Cycles", f"{total_cycles:,.0f}")
    col6.metric("Profit per MWh", f"${average_profit_per_mwh:,.0f}")

    # Find the date with the highest net revenue (most profitable day)
    most_profitable_day = daily_metrics['Net Revenue ($)'].idxmax()
    
    # Set the default selected date as the most profitable day
    selected_date = pd.to_datetime(st.date_input("Select a date", value=most_profitable_day))
    
    # Filter data for selected date and two adjacent days
    start_date = selected_date - pd.DateOffset(days=1)
    end_date = selected_date + pd.DateOffset(days=1)
    filtered_results = results_df[(results_df['Time'] >= start_date) & (results_df['Time'] <= end_date)]
    
    # Line chart
    st.line_chart(filtered_results[['Time', 'LMP $/MWh', 'SOC (MWh)']])
       
    # Display the metrics DataFrame as a table
    st.header("Dispatch Breakdown")
    st.table(metrics)
