##### Libraries

import pandas as pd
import matplotlib.pyplot as plt

##### CSV files

filepaths = {
    "Barcelona" : "C:\\Users\\Nicho\\Desktop\\Barcelona.csv",
    "Dubai" : "C:\\Users\\Nicho\\Desktop\\Dubai.csv",
    "London" : "C:\\Users\\Nicho\\Desktop\\London.csv",
    "Seattle" : "C:\\Users\\Nicho\\Desktop\\Seattle.csv",
    "Sydney" : "C:\\Users\\Nicho\\Desktop\\Sydney.csv"
}

Cities = {city: pd.read_csv(filepath) for city, filepath in filepaths.items()}

for city, df in Cities.items():
    print(f"{city} Dataset:")
    print(df.head())
    print(df.info(), "\n")


standard_columns = {
    "Price" : "Price",
    "Price_per_sqft" : "Price_per_SQFT",
    "Bedrooms" : "Bedrooms",
    "Bathrooms" : "Bathrooms",
    "SQFT" : "SQFT",
    "City" : "City",
    "Suburb" : "Suburb",
    "House Type" : "House_Type",
    "Built" : "Built",
    "Remodeled" : "Remodeled",
    "Country" : "Country",
    "Condition" : "Condition",
    "Zip Code" : "Zip_Code"
}

# Directly rename the 'Price_per_sqft' column to 'Price_per_SQFT' in the Dubai DataFrame
Cities['Dubai'].rename(columns={'Price_per_sqft': 'Price_per_SQFT'}, inplace=True)

# Assuming 'Cities' is your dictionary of DataFrames
for city, df in Cities.items():
    df.rename(columns={col: standard_columns.get(col.replace(" ", "_").replace("-", "_").lower(), col) for col in df.columns}, inplace=True)
    # Check if the renaming was successful
    print(f"{city} columns after renaming: {df.columns.tolist()}")

for df in Cities.values():
    # Ensure the column exists before trying to convert it
    if "Price_per_SQFT" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"].replace("[\$,]", "", regex=True), errors="coerce")
        df["Price_per_SQFT"] = pd.to_numeric(df["Price_per_SQFT"].replace("[\$,]", "", regex=True), errors="coerce")

# Correctly reference the Sydney DataFrame
Cities['Sydney']["Bedrooms"].fillna(Cities['Sydney']["Bedrooms"].mode()[0], inplace=True)
Cities['Sydney']["Bedrooms"] = Cities['Sydney']["Bedrooms"].astype(int)

DataFrame = pd.concat(Cities.values(), ignore_index=True)
DataFrame.to_csv("C:\\Users\\Nicho\\Desktop\\Global Housing Prices.csv", index=False)








