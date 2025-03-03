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
df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/wind.csv')

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])
df['WNDS'] = np.sqrt(df['10u']**2 + df['10v']**2)
# Group by year and month directly without setting 'time' as index
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month

# Calculate the mean for each group
monthly_avg = df.groupby(['year', 'month']).mean().reset_index()

# Rename columns for clarity
monthly_avg.rename(columns={'WNDS': 'monthly_avg_WNDS'}, inplace=True)

# Print the monthly average precipitation
print("Monthly average Temp per location:")
print(monthly_avg)

# Calculate the combined monthly average across both locations
combined_monthly_avg = df.groupby(['year', 'month'])['WNDS'].mean().reset_index()

# Rename columns for clarity
combined_monthly_avg.rename(columns={'WNDS': 'combined_avg_WNDS'}, inplace=True)

# Print the combined monthly average precipitation
print("Combined monthly average WNDS:")
print(combined_monthly_avg)

observed_df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/merged_WNDS.csv')

# Convert 'time' column to datetime
observed_df['time'] = pd.to_datetime(observed_df['Date'])
observed_df = observed_df.drop(columns=['Date'])
# Step 2: Calculate the monthly average for the observed data
observed_df['year'] = observed_df['time'].dt.year
observed_df['month'] = observed_df['time'].dt.month

# Group by 'year' and 'month' and calculate the mean of the observed data
monthly_observed_avg = observed_df.groupby(['year', 'month']).mean().reset_index()

# Rename the observed data column for clarity
monthly_observed_avg.rename(columns={'average_WNDS': 'monthly_avg_observed'}, inplace=True)  # Replace 'observed_value_column' with the actual column name in your observed data

# Step 3: Merge with the forecast data
merged_df = pd.merge(monthly_avg, monthly_observed_avg, on=['year', 'month'], how='inner')
merged_df['obs_WNDS'] = merged_df['monthly_avg_observed'] * 0.44704 #mph to m/s
merged_df['Forecast_WNDS'] = merged_df['monthly_avg_WNDS'] #m/s
# Print the merged DataFrame
print("Merged DataFrame:")
print(merged_df)

# Optionally, save the merged DataFrame to a CSV
merged_df.to_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/merged_ObsForecast_WNDS_data.csv', index=False)

# Define the date range
start_date = '2000-01-01'
end_date = '2024-12-31'

# Filter the DataFrame for the specified date period
merged_df_2000 = merged_df[(merged_df['time_x'] >= start_date) & (merged_df['time_x'] <= end_date)]

merged_df_clean = merged_df_2000.dropna(subset=['obs_WNDS', 'Forecast_WNDS'])

slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df_clean['obs_WNDS'], merged_df_clean['Forecast_WNDS'])
r_squared = r_value**2
print("R2_WNDS:", r_squared)
rmse = np.sqrt(mean_squared_error(merged_df_clean['obs_WNDS'], merged_df_clean['Forecast_WNDS']))
print(f"RMSE: {rmse:.2f} m/s")

# Plotting each ensemble member and the observed data
plt.figure(figsize=(12, 7))
plt.plot(merged_df_2000['time_x'], merged_df_2000['Forecast_WNDS'], label='Forecasted WNDS (m/s)', color='red', linewidth=1)

# Plot observed data separately
plt.plot(merged_df_2000['time_x'], merged_df_2000['obs_WNDS'], label='obs_WNDS (m/s)', color='blue', linewidth=2)

# Customize plot
plt.xlabel('Date')
plt.ylabel('obs_WNDS (m/s)')
plt.title('Observed vs Forecast obs_WNDS')
plt.legend()
plt.grid()
plt.show()

# Plot a x-y graph
# Scatter plot of observations vs. forecasts
plt.figure(figsize=(8, 8))
plt.scatter(
    merged_df_2000['obs_WNDS'], 
    merged_df_2000['Forecast_WNDS'], 
    alpha=0.6, color='blue', label='Data Points'
)

# Add a 1:1 line
max_value = max(merged_df_2000['obs_WNDS'].max(), merged_df_2000['Forecast_WNDS'].max())
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', linewidth=2, label='1:1 Line')

# Customize plot
plt.xlabel('Observed WNDS (m/s)')
plt.ylabel('Forecasted WNDS (m/s)')
plt.title('Observed vs. Forecasted WNDS with 1:1 Line')
plt.legend()
plt.grid()
# Set axis limits to start at 0, 0
plt.ylim(0, max_value)
plt.xlim(0, max_value)
plt.savefig('C:/Osama Tarabih/Forecast Bias Corrections/WND_Forecast_Obs_xy.png', dpi = 600)
# Show plot
plt.show()
