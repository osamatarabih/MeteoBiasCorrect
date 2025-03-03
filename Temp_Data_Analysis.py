# This Script verifies the Archived ECMWF rainfall data (tp) vs. average rainfall data in Lake Okeechobee
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.stats import gamma
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/2m_temperature.csv')

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Group by year and month directly without setting 'time' as index
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month

# Calculate the mean for each group
monthly_avg = df.groupby(['year', 'month']).mean().reset_index()

# Rename columns for clarity
monthly_avg.rename(columns={'2t': 'monthly_avg_temp'}, inplace=True)

# Print the monthly average precipitation
print("Monthly average Temp per location:")
print(monthly_avg)

# Calculate the combined monthly average across both locations
combined_monthly_avg = df.groupby(['year', 'month'])['2t'].mean().reset_index()

# Rename columns for clarity
combined_monthly_avg.rename(columns={'2t': 'combined_avg_temp'}, inplace=True)

# Print the combined monthly average precipitation
print("Combined monthly average Temp:")
print(combined_monthly_avg)

observed_df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/merged_AirT.csv')

# Convert 'time' column to datetime
observed_df['time'] = pd.to_datetime(observed_df['Date'])
observed_df = observed_df.drop(columns=['Date'])
# Step 2: Calculate the monthly average for the observed data
observed_df['year'] = observed_df['time'].dt.year
observed_df['month'] = observed_df['time'].dt.month

# Group by 'year' and 'month' and calculate the mean of the observed data
monthly_observed_avg = observed_df.groupby(['year', 'month']).mean().reset_index()

# Rename the observed data column for clarity
monthly_observed_avg.rename(columns={'average_AirT': 'monthly_avg_observed'}, inplace=True)  # Replace 'observed_value_column' with the actual column name in your observed data

# Step 3: Merge with the forecast data
merged_df = pd.merge(monthly_avg, monthly_observed_avg, on=['year', 'month'], how='inner')
merged_df['obs_Temp'] = merged_df['monthly_avg_observed'] #C
merged_df['Forecast_Temp'] = merged_df['monthly_avg_temp'] - 273.15 #K to C
# Print the merged DataFrame
print("Merged DataFrame:")
print(merged_df)

# Optionally, save the merged DataFrame to a CSV
merged_df.to_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/merged_ObsForecast_Temp_data.csv', index=False)

# Define the date range
start_date = '2000-01-01'
end_date = '2024-12-31'

# Filter the DataFrame for the specified date period
merged_df_2000 = merged_df[(merged_df['time_x'] >= start_date) & (merged_df['time_x'] <= end_date)]

merged_df_clean = merged_df_2000.dropna(subset=['obs_Temp', 'Forecast_Temp'])

slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df_clean['obs_Temp'], merged_df_clean['Forecast_Temp'])
r_squared = r_value**2
print("R2_temp:", r_squared)
rmse = np.sqrt(mean_squared_error(merged_df_clean['obs_Temp'], merged_df_clean['Forecast_Temp']))
print(f"RMSE: {rmse:.2f} C")

# Plotting each ensemble member and the observed data
plt.figure(figsize=(12, 7))
plt.plot(merged_df_2000['time_x'], merged_df_2000['Forecast_Temp'], label='Forecasted Temp (C)', color='red', linewidth=1)

# Plot observed data separately
plt.plot(merged_df_2000['time_x'], merged_df_2000['obs_Temp'], label='obs_Temp (C)', color='blue', linewidth=2)

# Customize plot
plt.xlabel('Date')
plt.ylabel('Temp (C)')
plt.title('Observed vs Forecast Temp')
plt.legend()
plt.grid()
plt.show()

# Plot a x-y graph
# Scatter plot of observations vs. forecasts
plt.figure(figsize=(8, 8))
plt.scatter(
    merged_df_2000['obs_Temp'], 
    merged_df_2000['Forecast_Temp'], 
    alpha=0.6, color='blue', label='Data Points'
)

# Add a 1:1 line
max_value = max(merged_df_2000['obs_Temp'].max(), merged_df_2000['Forecast_Temp'].max())
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', linewidth=2, label='1:1 Line')

# Customize plot
plt.xlabel('Observed Temp (C)')
plt.ylabel('Forecasted Temp (C)')
plt.title('Observed vs. Forecasted Temp with 1:1 Line')
plt.legend()
plt.grid()
# Set axis limits to start at 0, 0
plt.ylim(0, max_value)
plt.xlim(0, max_value)
plt.savefig('C:/Osama Tarabih/Forecast Bias Corrections/Temp_Forecast_Obs_xy.png', dpi = 600)
# Show plot
plt.show()
