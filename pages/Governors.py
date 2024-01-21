import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from page import page_config
import plotly.graph_objs as go


page_config()


st.header('Example Governor Page')

st.markdown('Simple Example of an attendance plot for each school in your trust. The data comes from the BI Connector.')

# Initialize connection.
conn = st.connection("snowflake")

# Query to get monthly averages
df_avg = conn.query("""
SELECT 
    application_id, 
    YEAR(date) AS year, 
    MONTH(date) AS month, 
    AVG(IFF(IS_PRESENT, 1, 0)) AS avg_true_proportion
FROM 
    ARBOR_BI_CONNECTOR_PRODUCTION.ARBOR_MIS_ENGLAND_MODELLED.ROLL_CALL_ATTENDANCE
WHERE 
    date >= DATEADD(year, -3, CURRENT_DATE())
GROUP BY 
    application_id, year, month
ORDER BY 
    application_id, year, month;
""")



# Combine YEAR and MONTH columns in averages dataframe to a datetime for proper sorting
df_avg['DATE'] = pd.to_datetime(df_avg[['YEAR', 'MONTH']].assign(DAY=1))

# Sort both dataframes
df_avg.sort_values(by='DATE', inplace=True)
# df_pred.sort_values(by='MONTH_YEAR', inplace=True)

# Get unique application IDs
unique_app_ids = df_avg['APPLICATION_ID'].unique()

# Create a separate Plotly figure for each application ID
for app_id in unique_app_ids:
    # Filter data for each application_id
    df_app_avg = df_avg[df_avg['APPLICATION_ID'] == app_id]
    # df_app_pred = df_pred[df_pred['APPLICATION_ID'] == app_id]

    # Create a new Plotly figure
    fig = go.Figure()

    # Add the monthly average line
    fig.add_trace(go.Scatter(
        x=df_app_avg['DATE'],
        y=df_app_avg['AVG_TRUE_PROPORTION'],
        mode='lines',
        name=f'Average'
    ))


    # Update layout if needed
    fig.update_layout(
        title=f'Attendance for {app_id}',
        xaxis_title='Date',
        yaxis_title='Proportion Present',
        legend_title='Legend',
    )

    # Plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)




