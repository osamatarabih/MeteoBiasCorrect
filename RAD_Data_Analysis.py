# This Script verifies the Archived ECMWF rainfall data (tp) vs. average rainfall data in Lake Okeechobee
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.stats import gamma
import matplotlib.pyplot as plt
from hydroeval import evaluator, kge

# Load the dataset
df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/SolarRad.csv')

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Group by year and month directly without setting 'time' as index
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month

# Calculate the mean for each group
monthly_avg = df.groupby(['year', 'month']).mean().reset_index()

# Rename columns for clarity
monthly_avg.rename(columns={'fdir': 'monthly_avg_Rad'}, inplace=True)

# Print the monthly average precipitation
print("Monthly average Rad per location:")
print(monthly_avg)

# Calculate the combined monthly average across both locations
combined_monthly_avg = df.groupby(['year', 'month'])['fdir'].mean().reset_index()

# Rename columns for clarity
combined_monthly_avg.rename(columns={'fdir': 'combined_avg_Rad'}, inplace=True)

# Print the combined monthly average precipitation
print("Combined monthly average Rad:")
print(combined_monthly_avg)

observed_df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/merged_RADT.csv')

# Convert 'time' column to datetime
observed_df['time'] = pd.to_datetime(observed_df['Date'])
observed_df = observed_df.drop(columns=['Date'])
# Step 2: Calculate the monthly average for the observed data
observed_df['year'] = observed_df['time'].dt.year
observed_df['month'] = observed_df['time'].dt.month

# Group by 'year' and 'month' and calculate the mean of the observed data
monthly_observed_avg = observed_df.groupby(['year', 'month']).mean().reset_index()

# Rename the observed data column for clarity
monthly_observed_avg.rename(columns={'average_RADT': 'monthly_avg_observed'}, inplace=True)  # Replace 'observed_value_column' with the actual column name in your observed data

# Step 3: Merge with the forecast data
merged_df = pd.merge(monthly_avg, monthly_observed_avg, on=['year', 'month'], how='inner')
merged_df['obs_Rad'] = merged_df['monthly_avg_observed'] #kW/m2
merged_df['Forecast_Rad'] = merged_df['monthly_avg_Rad']/3600000 #J/m2 to kW/m2
# Print the merged DataFrame
print("Merged DataFrame:")
print(merged_df)

# Optionally, save the merged DataFrame to a CSV
merged_df.to_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/merged_ObsForecast_Rad_data.csv', index=False)

# Define the date range
start_date = '2000-01-01'
end_date = '2024-12-31'

# Filter the DataFrame for the specified date period
merged_df_2000 = merged_df[(merged_df['time_x'] >= start_date) & (merged_df['time_x'] <= end_date)]

merged_df_clean = merged_df_2000.dropna(subset=['obs_Rad', 'Forecast_Rad'])

slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df_clean['obs_Rad'], merged_df_clean['Forecast_Rad'])
r_squared = r_value**2
print("R2_Rad:", r_squared)
rmse = np.sqrt(mean_squared_error(merged_df_clean['obs_Rad'], merged_df_clean['Forecast_Rad']))
print(f"RMSE: {rmse:.2f} kW/m2")
mae = np.mean(np.abs(merged_df_clean['obs_Rad'] - merged_df_clean['Forecast_Rad']))
print(f"MAE: {mae:.2f}")
kge_score = evaluator(kge, merged_df_clean['Forecast_Rad'], merged_df_clean['obs_Rad'])
print(f"KGE: {kge_score[0][0]:.4f}")


# Plotting each ensemble member and the observed data
plt.figure(figsize=(12, 7))
plt.plot(merged_df_2000['time_x'], merged_df_2000['Forecast_Rad'], label='Forecasted Rad (kW/m2)', color='red', linewidth=1)

# Plot observed data separately
plt.plot(merged_df_2000['time_x'], merged_df_2000['obs_Rad'], label='obs_Rad (kW/m2)', color='blue', linewidth=2)

# Customize plot
plt.xlabel('Date')
plt.ylabel('Rad (kW/m2)')
plt.title('Observed vs Forecast Rad')
plt.legend()
plt.grid()
plt.show()

# Plot a x-y graph
# Scatter plot of observations vs. forecasts
plt.figure(figsize=(8, 8))
plt.scatter(
    merged_df_2000['obs_Rad'], 
    merged_df_2000['Forecast_Rad'], 
    alpha=0.6, color='blue', label='Data Points'
)

# Add a 1:1 line
max_value = max(merged_df_2000['obs_Rad'].max(), merged_df_2000['Forecast_Rad'].max())
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', linewidth=2, label='1:1 Line')

# Customize plot
plt.xlabel('Observed Rad (kW/m2)')
plt.ylabel('Forecasted Rad (kW/m2)')
plt.title('Observed vs. Forecasted Rad with 1:1 Line')
plt.legend()
plt.grid()
# Set axis limits to start at 0, 0
plt.ylim(0, max_value)
plt.xlim(0, max_value)
plt.savefig('C:/Osama Tarabih/Forecast Bias Corrections/RAD_Forecast_Obs_xy.png', dpi = 600)

# Show plot
plt.show()
