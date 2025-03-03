import pandas as pd
import numpy as np
import os
import glob

path = 'C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/WNDS'  
all_files = glob.glob(os.path.join(path, "*.csv"))
# List to store dataframes for merging
dfs = []

# Loop through files, read, and process each
for file in all_files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Ensure 'date' is datetime type for easier merging
    df['Date'] = pd.to_datetime(df['Daily Date'])

    df.drop(columns=['Daily Date'], inplace=True)  # Optionally drop original date column

    # Rename the rainfall column to avoid conflicts when merging
    df.rename(columns={'Data Value': f'WNDS_{len(dfs)}'}, inplace=True)

    # Append the dataframe to the list
    dfs.append(df[['Date', f'WNDS_{len(dfs)}']])

# Merge all dataframes on the 'date' column
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = merged_df.merge(df, on='Date', how='outer')

# Calculate the average rainfall, ignoring NaN values
merged_df['average_WNDS'] = merged_df.filter(like='WNDS').mean(axis=1)

# Display the resulting dataframe
print(merged_df)

# Save to CSV if needed
merged_df.to_csv('C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/WNDS/merged_WNDS.csv', index=False)

# assemble to monthly average
merged_df.set_index('Date', inplace=True)
merged_df.resample('M').mean().to_csv('C:/Osama Tarabih/Forecast Bias Corrections/DBHYDRO_Data/monthly_WNDS.csv')