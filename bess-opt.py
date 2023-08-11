import streamlit as st
import requests
from pulp import *
import gridstatusio
from gridstatusio import GridStatusClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import base64

#Streamlit Setup

#width
st.set_page_config(layout="centered")

#hide menu bars
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

st.markdown(hide_default_format, unsafe_allow_html=True)

# Function to fetch solar output from PV Watts

def fetch_solar_output(api_key, address, system_capacity, dc_ac_ratio, module_type, array_type, tilt, azimuth, losses):
    pvwatts_base_url = "https://developer.nrel.gov/api/pvwatts/v8"
    endpoint = f"{pvwatts_base_url}.json"

    params = {
        "api_key": api_key,
        "address": address,
        "system_capacity": system_capacity,
        "dc_ac_ratio": dc_ac_ratio,
        "module_type": module_type,
        "array_type": array_type,
        "tilt": tilt,
        "azimuth": azimuth,
        "losses": losses,
        "timeframe": "hourly",
    }

    # Print the generated API URL for debugging
    generated_url = f"{endpoint}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"
    st.write(f"Generated API URL: {generated_url}")

    response = requests.get(endpoint, params=params)

    # Print debugging information
    st.write("Status Code:", response.status_code)
    st.write("Headers:", response.headers)
    st.write("Response Content:", response.text)
    
    if response.status_code != 200:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return None

    data = response.json()
    data["outputs"]["ac"] = [x / 1000000 for x in data["outputs"]["ac"]] # convert to MW from W
    return data["outputs"]["ac"]

# Title
st.title('CACS-Opt')

# Header
st.header('Inputs')

# Site Information
with st.expander("🗺️ Site Information"):
    col1, col2 = st.columns(2)
    with col1:
           address = st.text_input('Site Address', 'Number Street, State Zip')
    with col2:
           utility = st.radio("Utility",('PG&E', 'SCE'))

# Solar System
with st.expander("☀️ Solar System"):
    system_capacity = st.number_input('Solar Capacity (kW-DC)')
    col3, col4 = st.columns(2)
    with col3:
        dc_ac_ratio = st.number_input('DC-AC Ratio')
        array_type_selection = st.radio("Array Type", ('Fixed - Open Rack', 'Fixed - Roof Mounted', 'Single Axis Tracker'))
        if array_type_selection == 'Fixed - Open Rack': 
            array_type = 0
        elif array_type_selection == 'Fixed - Roof Mounted': 
            array_type = 1
        elif array_type_selection == 'Single Axis Tracker': 
            array_type = 2
        tilt = st.number_input('Tilt', value=10)
    with col4:
        losses = st.number_input('Losses %', value=14)
        module_type_selection = st.radio("Module Type",('Standard', 'Premium', 'Thin film'))
        if module_type_selection == 'Standard': 
            module_type = 0
        elif module_type_selection == 'Premium': 
            module_type = 1
        elif module_type_selection == 'Thin film': 
            module_type = 2
        azimuth = st.number_input('Azimuth', value=180)

# Battery System
with st.expander("🔋 Battery System"):
    col5, col6 = st.columns(2)
    with col5:
        energy_capacity = st.number_input("Energy capacity (MWh)", min_value=0.0, max_value=1000.0, value=100.0)
        charge_power_limit = st.number_input("Charge power limit (MW)", min_value=0.0, value=25.0)
        discharge_power_limit = st.number_input("Discharge power limit (MW)", min_value=0.0, value=25.0)
        SOC_initial = st.number_input("Initial SOC (MWh)", min_value=0.0, value=0.0)
        daily_cycle_limit = st.number_input("Daily cycle limit", min_value=0.0, max_value=10.0, value=1.0)
    with col6:
        discharge_efficiency = st.number_input("Discharge efficiency", min_value=0.0, max_value=1.0, value=0.95)
        charge_efficiency = st.number_input("Charge efficiency", min_value=0.0, max_value=1.0, value=0.95)
        SOC_max = st.number_input("Max SOC (MWh)", min_value=0.0, value=100.0)
        SOC_min = st.number_input("Min SOC (MWh)", min_value=0.0, value=0.0)
        annual_cycle_limit = st.number_input("Annual cycle limit", min_value=0.0, value=300.0)

# Initialize variables
results_df = None
chart_data = None

# Button to run the optimization
if st.button('Run Optimization'):

    # Outputs Header
    st.header('Outputs')
    
    # Make it rain
    from streamlit_extras.let_it_rain import rain
    rain(
        emoji="🔋",
        font_size=54,
        falling_speed=3,
        animation_length=1,
    )

    st.info('Optimization started', icon="🤯")
    
    # Create dataframe for relevant columns and extract prices as a list from it
    if utility == 'PG&E':
        da_prices_df = pd.read_csv('pge_rates.csv')
    elif utility == 'SCE':
        da_prices_df = pd.read_csv('sce_rates.csv')

    da_prices_df = da_prices_df.rename(columns={'date-time': 'interval_start_local', '$/kWh': 'lmp'})
    da_prices = da_prices_df['lmp'].tolist()

    # Price Forecast for num_hours hours
    num_hours = len(da_prices)
    num_days = num_hours / 24
    total_cycle_limit = (num_days / 365) * annual_cycle_limit

    # Fetch solar output from PV Watts
    api_key = "7ENvpt1oAXJkRb56AtQOPttQJQJm5nF5lyeMkxXe"
    solar_output = fetch_solar_output(api_key, address, system_capacity, dc_ac_ratio, module_type, array_type, tilt, azimuth, losses) 
                                      
    # Create a function to define the optimization model
    def optimization_model(num_hours, da_prices):
        # Variables
        charge_vars = LpVariable.dicts("Charging", range(num_hours), lowBound=0, upBound=charge_power_limit)
        discharge_vars = LpVariable.dicts("Discharging", range(num_hours), lowBound=0, upBound=discharge_power_limit)
        SOC_vars = LpVariable.dicts("SOC", range(num_hours + 1), lowBound=SOC_min, upBound=SOC_max)  # Including initial SOC

        # Problem
        prob = LpProblem("Battery Scheduling", LpMaximize)

        # Objective function
        prob += lpSum([da_prices[t] * (solar_output[t] + discharge_efficiency * discharge_vars[t] - charge_vars[t] / charge_efficiency) for t in range(num_hours)])

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

    st.success('Optimization complete!', icon="✅")
    
    # Prepare data for results table
    results = []
    for t in range(num_hours):
     results.append([da_prices_df['interval_start_local'][t], da_prices_df['lmp'][t], charge_vars[t].varValue, discharge_vars[t].varValue, SOC_vars[t].varValue])
    
    results_df = pd.DataFrame(results, columns=["Time", "LMP $/MWh", "Charging (MW)", "Discharging (MW)", "SOC (MWh)"])

# Calculate hourly metrics
if results_df is not None:
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
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Discharging Revenue", f"${total_discharging_revenue:,.0f}")
    col2.metric("Total Charging Costs", f"${total_charging_costs:,.0f}")
    col3.metric("Total Net Revenue", f"${total_net_revenue:,.0f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Days Analyzed", f"{num_days:,.0f}")
    col5.metric("Total Cycles", f"{total_cycles:,.0f}")
    col6.metric("Profit per MWh", f"${average_profit_per_mwh:,.0f}")

    #Header
    st.header("Dispatch Chart")
       
    # Find the date with the highest net revenue (most profitable day)
    most_profitable_day = daily_metrics['Net Revenue ($)'].idxmax()
    
    # Set the default selected date as the most profitable day
    selected_date = pd.to_datetime(st.date_input("Select a date", value=most_profitable_day))
    
    # Convert the selected date to the same time zone as the 'results_df' data
    selected_date = selected_date.tz_localize('US/Pacific')
    
    # Filter data for selected date and two adjacent days
    start_date = selected_date - pd.DateOffset(days=1)
    end_date = selected_date + pd.DateOffset(days=1)
    filtered_results = results_df[(results_df['Time'] >= start_date) & (results_df['Time'] <= end_date)]
    
    # Prepare data for line chart
    chart_data = filtered_results[['Time', 'LMP $/MWh', 'SOC (MWh)']].copy()
    chart_data['Time'] = pd.to_datetime(chart_data['Time']).dt.tz_localize(None)
    
    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add the first line trace (LMP $/MWh) to the figure
    fig.add_trace(
        go.Scatter(x=chart_data['Time'], y=chart_data['LMP $/MWh'], name='LMP $/MWh'),
        secondary_y=False
    )
    
    # Add the second line trace (SOC MWh) to the figure
    fig.add_trace(
        go.Scatter(x=chart_data['Time'], y=chart_data['SOC (MWh)'], name='SOC (MWh)'),
        secondary_y=True
    )
    
    # Update the layout of the figure
    fig.update_layout(
        xaxis=dict(title='Time'),
        yaxis=dict(title='LMP $/MWh', side='left', showgrid=False),
        yaxis2=dict(title='SOC (MWh)', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Line chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the metrics DataFrame as a table
    st.header("Performance Summary")
    st.table(metrics)

    # Download button for results_df CSV
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    button_text = "Download Results CSV"
    button_label = f"Download {len(results_df)} rows"
    st.download_button(button_text, data=b64, file_name='results.csv', mime='text/csv')
