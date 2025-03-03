import pandas as pd
import numpy as np
import os
import glob

def process_and_save_data(path, prefix):
    """
    Processes CSV files in the given path, merges them, calculates averages, and saves the results.
    
    Parameters:
    - path: str, path to the folder containing the CSV files.
    - prefix: str, prefix for the variable (e.g., 'AirT', 'RADT', 'WNDS').
    """
    # Get all CSV files in the directory
    all_files = glob.glob(os.path.join(path, "*.csv"))
    dfs = []

    # Loop through files, read, and process each
    for file in all_files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Daily Date'])  # Convert date column to datetime
        df.drop(columns=['Daily Date'], inplace=True)  # Drop the original date column
        df.rename(columns={'Data Value': f'{prefix}_{len(dfs)}'}, inplace=True)  # Rename data value column
        dfs.append(df[['Date', f'{prefix}_{len(dfs)}']])  # Append processed dataframe

    # Merge all dataframes on the 'Date' column
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='Date', how='outer')

    # Calculate the average, ignoring NaN values
    merged_df[f'average_{prefix}'] = merged_df.filter(like=prefix).mean(axis=1)

    # Save merged daily data
    merged_daily_path = os.path.join(path, f'merged_{prefix}.csv')
    merged_df.to_csv(merged_daily_path, index=False)
    print(f"Saved merged daily data to {merged_daily_path}")

    # Assemble monthly averages and save
    merged_df.set_index('Date', inplace=True)
    monthly_avg_path = os.path.join(path, f'monthly_{prefix}.csv')
    merged_df.resample('M').mean().to_csv(monthly_avg_path)
    print(f"Saved monthly averages to {monthly_avg_path}")

# Define paths and prefixes
variables = {
    'AirT': 'C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/AirT',
    'RADT': 'C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/RADT',
    'WNDS': 'C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/WNDS'
}

# Process each variable
for prefix, path in variables.items():
    process_and_save_data(path, prefix)
