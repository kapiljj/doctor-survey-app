import pandas as pd

# Load the dataset
df = pd.read_csv('doctors_data.csv')

# Preprocess the data
df['Login Time'] = pd.to_datetime(df['Login Time'])
df['Logout Time'] = pd.to_datetime(df['Logout Time'])
df['Active Hours'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 3600

# Save the preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
