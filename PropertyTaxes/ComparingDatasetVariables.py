# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:02:49 2024

@author: richard.feir
"""
# In[]

import pandas as pd
import numpy as np

# Reading the CSV files
StateGDP = pd.read_csv("StateGDP.csv")
StateGovFinances1998to2021 = pd.read_csv("StateGovFinances1998to2021.csv")
StateGovFinances2002to2021 = pd.read_csv("StateGovFinances2002to2021.csv")
StateGovFinances2005to2021 = pd.read_csv("StateGovFinances2005to2021.csv")
StateGovFinances2009to2021 = pd.read_csv("StateGovFinances2009to2021.csv")
StateGovFinancesTo2008 = pd.read_csv("StateGovFinancesTo2008.csv")

# Preparing the StateGovFinancesTo2008 dataset
StateGovFinancesTo2008['State'] = StateGovFinancesTo2008['Name'].str[:2]
StateGovFinancesTo2008.replace(-11111, np.nan, inplace=True)

# Correcting Year column to numeric in StateGovFinances1998to2021
StateGovFinances1998to2021['Year'] = StateGovFinances1998to2021['Year'].str[:4]

# Function to convert 'Year' column to numeric
def convert_year_to_numeric(df, year_col='Year'):
    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')  # Convert to numeric, coerce errors to NaN
    else:
        print(f"{year_col} column not found in DataFrame")
    return df

# Apply the function to both datasets
StateGovFinances1998to2021 = convert_year_to_numeric(StateGovFinances1998to2021)
StateGovFinancesTo2008 = convert_year_to_numeric(StateGovFinancesTo2008, year_col='Year4')

# Renaming 'Year4' to 'Year'
StateGovFinancesTo2008 = StateGovFinancesTo2008.rename(columns={'Year4': 'Year'})

# Filtering the datasets
StateGovFinances1998to2021 = StateGovFinances1998to2021[
    (StateGovFinances1998to2021['Format'] == 'State government amount') & 
    (StateGovFinances1998to2021['Year'] <= 2008)
]

StateGovFinancesTo2008 = StateGovFinancesTo2008[
    (StateGovFinancesTo2008['Year'] >= 1998)
]

# In[]

# Merging the datasets on 'State' and 'Year'
merged_data = pd.merge(StateGovFinances1998to2021, StateGovFinancesTo2008, on=['State', 'Year'], how='inner')


# In[]

# Set pandas option to display all info columns
pd.set_option('display.max_info_columns', 500)  # Set to a large enough number to display all columns

# Function to convert columns to float64, handling null values
def convert_to_float_with_null(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Identify columns that are currently objects
object_columns = merged_data.select_dtypes(include=['object']).columns

# Convert object columns to float64 with null values
merged_data = convert_to_float_with_null(merged_data, object_columns)


# In[]

# Calculating correlation matrices for each year in the range 1998 to 2008
for year in range(1998, 2008 + 1):
    yearly_data = merged_data[merged_data['Year'] == year]
    if not yearly_data.empty:
        # Select numeric columns for the correlation matrix calculation
        numeric_data = yearly_data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        print(f"Correlation matrix for {year}:")
        print(corr_matrix)
        print("\n")
        
# In[]

# Find pairs where correlation coefficient is exactly 1
correlation_pairs = []
cols = corr_matrix.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if corr_matrix.iloc[i, j] == 1.0 and cols[i] != cols[j]:
            correlation_pairs.append((cols[i], cols[j]))

# Print the results
print("Highly correlated pairs with correlation coefficient of 1:")
for pair in correlation_pairs:
    print(pair)