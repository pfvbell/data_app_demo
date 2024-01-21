import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os


st.header("Model Selection Demo")

# Initialize connection.
conn = st.connection("snowflake")


df_avg = conn.query("""
SELECT * FROM DEMO_LOAD_DATA.PUBLIC.ATTENDANCE_MONTHLY_AVERAGE_MAT
""")

# Query to get monthly averages
pandas_df = conn.query("""
SELECT * FROM DEMO_LOAD_DATA.PUBLIC.ATTENDANCE_MONTHLY_AVERAGE_MAT
""")

pandas_df['DATE'] = pd.to_datetime(pandas_df[['YEAR', 'MONTH']].assign(DAY=1))
pandas_df.sort_values(['APPLICATION_ID', 'DATE'], inplace=True)
pandas_df['LAG_1'] = pandas_df.groupby('APPLICATION_ID')['AVG_TRUE_PROPORTION'].shift(1)
pandas_df.dropna(inplace=True)

num_months = st.slider('How many months of training data?', 12, 36)

# Filter dataframe based on user's input
latest_date = pandas_df['DATE'].max()
earliest_date = latest_date - pd.DateOffset(months=num_months)
filtered_df = pandas_df[pandas_df['DATE'] >= earliest_date]

st.markdown('**Which Models would you like to run?**')
lr_checked = st.checkbox('Linear Regression')
rf_checked = st.checkbox('Random Forest')
lgbm_checked = st.checkbox('LGBM')

future_predictions = []
model_mse = []

run_button = st.button('Run Models')

if run_button:
    try:
        for app_id in filtered_df['APPLICATION_ID'].unique():
            app_df = filtered_df[filtered_df['APPLICATION_ID'] == app_id]
            X = app_df[['LAG_1']]
            y = app_df['AVG_TRUE_PROPORTION']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Prepare X for OLS (it requires a constant term for intercept)
            X_train_ols = sm.add_constant(X_train)
            X_test_ols = sm.add_constant(X_test)
        
            # Train RandomForestRegressor if selected
            if rf_checked:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred_test = rf_model.predict(X_test)
                rf_mse = mean_squared_error(y_test, rf_pred_test)
                model_mse.append({'application_id': app_id, 'model': 'RandomForest', 'mse': rf_mse})
        
            # Train LGBMRegressor if selected
            if lgbm_checked:
                lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
                lgbm_model.fit(X_train, y_train)
                lgbm_pred_test = lgbm_model.predict(X_test)
                lgbm_mse = mean_squared_error(y_test, lgbm_pred_test)
                model_mse.append({'application_id': app_id, 'model': 'LGBM', 'mse': lgbm_mse})
        
            # Train OLS if selected
            if lr_checked:
                ols_model = sm.OLS(y_train, X_train_ols).fit()
                ols_pred_test = ols_model.predict(X_test_ols)
                ols_mse = mean_squared_error(y_test, ols_pred_test)
                model_mse.append({'application_id': app_id, 'model': 'OLS', 'mse': ols_mse})
        
            # Predict future values if any model is trained
            if rf_checked or lgbm_checked or lr_checked:
                last_known_value = app_df.iloc[-1]['LAG_1']
                last_date = app_df.iloc[-1]['DATE']
                
                # Predict the next 6 months
                for i in range(1, 7):
                    next_month_date = last_date + pd.DateOffset(months=i)
                    month_year = next_month_date.strftime('%Y-%m')
        
                    # Make predictions using each model
                    if rf_checked:
                        rf_pred = rf_model.predict([[last_known_value]])[0]
                        future_predictions.append({
                            'application_id': app_id,
                            'date': month_year,
                            'model': 'RandomForest',
                            'prediction': rf_pred,
                            'mse': rf_mse
                        })
        
                    if lgbm_checked:
                        lgbm_pred = lgbm_model.predict([[last_known_value]])[0]
                        future_predictions.append({
                            'application_id': app_id,
                            'date': month_year,
                            'model': 'LGBM',
                            'prediction': lgbm_pred,
                            'mse': lgbm_mse
                        })
        
                    if lr_checked:
                        ols_pred = ols_model.predict(sm.add_constant(np.array([[1, last_known_value]])))[0]
                        future_predictions.append({
                            'application_id': app_id,
                            'date': month_year,
                            'model': 'OLS',
                            'prediction': ols_pred,
                            'mse': ols_mse
                        })
        
                    # Update last_known_value for the next prediction
                    preds = []
                    if rf_checked:
                        preds.append(rf_pred)
                    if lgbm_checked:
                        preds.append(lgbm_pred)
                    if lr_checked:
                        preds.append(ols_pred)
                    
                    if preds:
                        last_known_value = sum(preds) / len(preds)  # Averaging predictions for the next step
        
        # Create a DataFrame with the future predictions
        df_pred = pd.DataFrame(future_predictions)
        st.dataframe(df_pred)
        
        # Convert all column names to lowercase for both dataframes
        df_avg.columns = df_avg.columns.str.lower()
        df_pred.columns = df_pred.columns.str.lower()
        
        # Convert 'date' in df_pred to datetime for sorting, if not already in datetime format
        df_pred['date'] = pd.to_datetime(df_pred['date'])
        
        # Combine 'year' and 'month' columns in df_avg to a datetime for sorting
        df_avg['date'] = pd.to_datetime(df_avg[['year', 'month']].assign(day=1))
        
        # Sort both dataframes by 'date'
        df_avg.sort_values(by='date', inplace=True)
        df_pred.sort_values(by='date', inplace=True)
        
        # Get unique application IDs
        unique_app_ids = df_avg['application_id'].unique()
        
        # Define a color map for the models
        model_colors = {
            'RandomForest': 'blue',
            'LGBM': 'green',
            'OLS': 'red'
        }
        
        # Create a separate Plotly figure for each application ID
        for app_id in unique_app_ids:
            # Filter data for each application_id
            df_app_avg = df_avg[df_avg['application_id'] == app_id]
            df_app_pred = df_pred[df_pred['application_id'] == app_id]
        
            # Create a new Plotly figure
            fig = go.Figure()
            
            # Add the monthly average line
            fig.add_trace(go.Scatter(
                x=df_app_avg['date'],
                y=df_app_avg['avg_true_proportion'],
                mode='lines',
                name='Average Attendance',
                line=dict(color='black')
            ))
            
            # Plot predictions for each model
            mse_values = {}
            for model in model_colors.keys():
                model_df = df_app_pred[df_app_pred['model'] == model]
                if not model_df.empty:
                    mse_values[model] = model_df['mse'].iloc[0]  # Assuming MSE is constant for all predictions
                    fig.add_trace(go.Scatter(
                        x=model_df['date'],
                        y=model_df['prediction'],
                        mode='lines',
                        name=f'{model} Prediction',
                        line=dict(color=model_colors[model], dash='dot')
                    ))
        
            # Update layout
            fig.update_layout(
                title=f'Attendance for Application ID: {app_id}',
                xaxis_title='Date',
                yaxis_title='Attendance Proportion',
                legend_title='Type'
            )
            
            # Plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the MSE values for each model
            if mse_values:
                st.markdown('**Mean Squared Error:**')
                for model, mse in mse_values.items():
                    st.markdown(f"**{model}:** {mse:.5f}")
    except:
        st.info('Try Different Parameters')



    
    

