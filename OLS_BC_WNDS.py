import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import percentileofscore, scoreatpercentile
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def ols_combination_bias_correction(observed, forecast):
    """
    Perform OLS combination bias correction on forecast data using observed data.
    
    Parameters:
    observed (array-like): Array of observed WNDS data.
    forecast (array-like): Array of forecasted WNDS data.
    
    Returns:
    corrected_forecast (array-like): Bias-corrected forecast data.
    """
    # Ensure the inputs are numpy arrays
    observed = np.array(observed)
    forecast = np.array(forecast)
    
    # Add a constant term to the forecast data for the OLS regression
    forecast_with_constant = sm.add_constant(forecast)
    
    # Fit the OLS model
    model = sm.OLS(observed, forecast_with_constant).fit()
    # Extract parameters
    intercept, slope = model.params

    # Get the corrected forecast using the OLS model parameters
    corrected_forecast = model.predict(forecast_with_constant)
    
    print(f"Slope (m): {slope:.4f}")
    print(f"Intercept (b): {intercept:.4f}")
    print(f"OLS Regression Equation: Corrected Forecast = {slope:.4f} * Forecast + {intercept:.4f}")
    
    return corrected_forecast, slope, intercept

merged_df = pd.read_csv('C:/Osama Tarabih/Forecast Bias Corrections/ECMWF_Data/merged_ObsForecast_WNDS_data.csv')

# Define the date range
start_date = '2000-01-01'
end_date = '2024-12-31'

# Filter the DataFrame for the specified date period
merged_df_2000 = merged_df[(merged_df['time_x'] >= start_date) & (merged_df['time_x'] <= end_date)]

merged_df_clean = merged_df_2000.dropna(subset=['obs_WNDS', 'Forecast_WNDS'])

observed_WNDS = merged_df_clean['obs_WNDS']
forecast_WNDS = merged_df_clean['Forecast_WNDS']

# Call the modified function
corrected_forecast_WNDS, slope, intercept = ols_combination_bias_correction(observed_WNDS, forecast_WNDS)

# Add corrected data to the DataFrame
merged_df_clean['forecasted_BC'] = corrected_forecast_WNDS

# Print the slope and intercept explicitly
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df_clean['obs_WNDS'], merged_df_clean['forecasted_BC'])
r_squared = r_value**2
print("R2_WNDS:", r_squared)
rmse = np.sqrt(mean_squared_error(merged_df_clean['obs_WNDS'], merged_df_clean['forecasted_BC']))
print(f"RMSE: {rmse:.2f} m/s")

# Plotting each ensemble member and the observed data
plt.figure(figsize=(12, 7))
plt.plot(merged_df_clean['time_x'], merged_df_clean['Forecast_WNDS'], label='Forecasted WNDS (m/s)', color='red', linewidth=1)

# Plot observed data separately
plt.plot(merged_df_clean['time_x'], merged_df_clean['obs_WNDS'], label='Observed WNDS (m/s)', color='blue', linewidth=2)
# plot bias corrected data
plt.plot(merged_df_clean['time_x'], merged_df_clean['forecasted_BC'], label='bias corrected WNDS', color='green', linewidth=1)

# Customize plot
plt.xlabel('Date')
plt.ylabel('WNDS (m/s)')
plt.title('Observed vs Forecast WNDS')
plt.legend()
plt.grid()
plt.xticks(merged_df_clean['time_x'][::12], rotation=45)
plt.savefig('C:/Osama Tarabih/Forecast Bias Corrections/WNDS_BC_OLS.png', dpi = 600)
plt.show()
