##### Nicholas Catani
##### ---------------
##### Stock Market Analysis
##### ---------------
##### Seattle, WA, 11/02/2023


### LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import geopandas as gpd
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

### CSV

df = pd.read_csv(r"C:\Users\Nicho\Desktop\WSP.csv")

### DATA ALTERATIONS

df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").apply(lambda x: x.replace(tzinfo=pytz.UTC))

### EXPLORATORY DATA ANALYSIS

# # Display the first few rows of the dataset
# print("First 5 rows of the dataset:")
# print(df.head())
#
# # Get a summary of the dataset
# print("\nSummary of the dataset:")
# print(df.info())
#
# # Get basic statistics of the numeric columns
# print("\nSummary statistics of numeric columns:")
# print(df.describe())
#
# # Check for missing values
# print("\nMissing values in the dataset:")
# print(df.isnull().sum())


######## How many stocks have shown a closing price between $50 and $150 in the past month?

# # Calculate the date one month ago from "2023-09-20"
# target_date = datetime(2023, 9, 20, tzinfo=pytz.UTC)
# one_month_ago = target_date - timedelta(days=30)
#
# # Filter and count stock closing prices between $50 and $150 one month ago
# filtered_data = df[(df["Date"] >= one_month_ago) & (df["Date"] < target_date) & (df["Close"] >=50) & (df["Close"] <= 150)]
# count = len(filtered_data)
# selected_columns = ["Brand_Name", "Close", "Volume", "Country"]
# filtered_data = filtered_data[selected_columns]
# print(f"Number of stock closing prices between $50 and $150 one month ago from {target_date}: {count}")
# print(filtered_data)
#
# # Group the stocks by country and count the # of stocks in each country
# country_counts = filtered_data["Country"].value_counts()
#
# # Create a bar chart
# fig, ax = plt.subplots(figsize=(10, 6))
# bars = ax.bar(country_counts.index, country_counts.values, color='red')
# ax.set_xlabel('Country')
# ax.set_ylabel('Number of Stocks')
# ax.set_title('Stocks by Country')
# for bar, count in zip(bars, country_counts.values):
#     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count), ha="center")
# plt.show()


######## How did the Tech industry's perform in the last quarter?

# # Filter for the first 10 tech companies headquartered in the USA
# tech_usa_data = df[(df["Country"] == "usa") & (df["Industry_Tag"] == "technology")].groupby("Ticker").head(90)
#
# # Sort the data by "Date" in ascending order
# tech_usa_data = tech_usa_data.sort_values(by="Date", ascending=True)
# end_date = datetime(2023, 9, 20, tzinfo=pytz.UTC)
# start_date = end_date - timedelta(days=90)
#
# # Filter the data for the last quarter
# last_quarter_data = tech_usa_data[(tech_usa_data["Date"] >= start_date) & (tech_usa_data["Date"] <= end_date)]
#
# # Calculate the stock performance for each company in the last quarter
# stock_performance = last_quarter_data.groupby("Ticker").apply(lambda x: (x["Close"].iloc[0] - x["Close"].iloc[-1]) / x["Close"].iloc[0])
# print(stock_performance)
#
# # Plot the stock performance trend for each company in the last quarter
# plt.figure(figsize=(12, 6))
# for ticker, group in last_quarter_data.groupby("Ticker"):
#     plt.plot(group["Date"], group["Close"], label=ticker)
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.title('Stock Performance Trend for the First 10 Companies in the Last Quarter')
# plt.legend(loc='best')
# plt.grid()
# plt.show()


######## Which sector performed well in the last six months?

# # Ensure the data is sorted by 'Date'
# data = df.sort_values(by=['Date', "Industry_Tag"])
#
# # List of the specific industries you want to extract
# selected_industries = ["automotive", "technology", "finance", "hospitality"]
#
# # Remove duplicates by aggregating the data within each group
# data = data.groupby(['Date', 'Industry_Tag']).agg({'Close': 'mean'}).reset_index()
#
# # Filter the DataFrame to include only the selected industries
# filtered_data = data[data['Industry_Tag'].isin(selected_industries)]
#
# # Calculate the six-month return for each selected industry
# filtered_data['Return'] = filtered_data.groupby('Industry_Tag')['Close'].pct_change(periods=6) * 100
#
# # Pivot the data for plotting
# pivoted_data = filtered_data.pivot(index='Date', columns='Industry_Tag', values='Return')
#
# # Plot the performance of each selected industry with six-month returns
# plt.figure(figsize=(12, 6))
# for industry in selected_industries:
#     plt.plot(pivoted_data.index, pivoted_data[industry], label=industry)
# plt.xlabel('Date')
# plt.ylabel('Return (%)')
# plt.title('Selected Industry Performance in the Past Years')
# plt.legend(loc='best')
# plt.grid()
# plt.show()


######## Percentage of Companies per Industry

# # Remove duplicates by aggregating the data
# data = df.drop_duplicates()
#
# # Group the data by 'Industry_Tag' and count the number of companies in each group
# industry_counts = data['Industry_Tag'].value_counts()
#
# # Calculate the total number of companies
# total_companies = len(data)
#
# # Filter industries with a percentage of companies greater than or equal to 3%
# filtered_industries = industry_counts[industry_counts / total_companies >= 0.03]
#
# # Create a pie chart with the filtered industries
# plt.figure(figsize=(8, 8))
# plt.pie(filtered_industries, labels=filtered_industries.index, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
#
# plt.title('Percentage of Companies per Industry\n')
# plt.show()


######## Total number of Companies per Country ###### Requires more work!

# Remove duplicates by aggregating the data to count the number of unique Brand_Names per Country
country_brand_counts = df.drop_duplicates(subset=['Country', 'Brand_Name']).groupby('Country')['Brand_Name'].count().reset_index()
country_brand_counts.columns = ['Country', 'Brand_Name Count']

# Display the result
print(country_brand_counts)

# Load a world map dataset from geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the world map dataset with the company counts
world = world.merge(country_brand_counts, left_on='iso_a3', right_on='Country', how='left')

# Plot the cartogram
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax, linewidth=1)
world.plot(column='Brand_Name Count', ax=ax, cmap="viridis", legend=True, legend_kwds={'label': "Number of Brands per Country"})
plt.title('Cartogram of Number of Brands per Country')
plt.show()

######### Shortcut!

# # Remove duplicates by counting the number of unique Brand_Names per Country
# country_brand_counts = df.drop_duplicates(subset=['Country', 'Brand_Name']).groupby('Country')['Brand_Name'].count().reset_index()
#
# # Create a basic cartogram
# fig = px.choropleth(country_brand_counts, locations="Country", locationmode="country names", color="Brand_Name",
#                     hover_name="Country", color_continuous_scale='viridis',
#                     title='Total Number of Brands per Country')
# fig.show()



### TRAINING SESSION


### First training with linear regression

# # Define the features and target variable
# features = ["Stock Splits", "Dividends", "Industry_Tag"]
# target = "Close"
#
# # Preprocessing for categorical features
# categorical_features = ["Industry_Tag"]
# categorical_transformer = Pipeline(steps=[
#     ("onehot", OneHotEncoder(handle_unknown="ignore"))
# ])
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("cat", categorical_transformer, categorical_features)
#     ])
#
# # Split the dataset into training and testing sets using cross-validation
# X = df[features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Create a pipeline with preprocessing and linear regression model
# model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("regressor", LinearRegression())
# ])
#
# # Fit the model using cross-validation
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
#
# # Print the mean and standard deviation of the cross-validation scores
# print("Cross-Validation Scores:")
# print("Mean:", round(-scores.mean(), 2))
# print("Standard Deviation:", round(scores.std(), 2))
#
# # Fit the model on the entire training data
# model.fit(X_train, y_train)
#
# # Make predictions on the testing data
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# # Print the evaluation metrics
# print(f"\nMean Absolute Error: {round(mae, 2)}")
# print(f"Mean Squared Error: {round(mse, 2)}")
# print(f"R-squared: {round(r2, 2)}")

##### Comments #####
# Overall, the results indicate that the linear regression model, as currently configured,
# has limited predictive power, as indicated by the relatively high MAE, MSE, and low
# R-squared value. It is possible that the model may benefit from additional
# feature engineering, more complex modeling techniques, or different algorithms
# to improve its predictive performance.


### Second training with random forest


# # Define the features and target variable
features = ["Stock Splits", "Dividends", "Volume"]
target = "Close"

# Split the dataset into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error: {round(mae, 2)}")
print(f"Mean Squared Error: {round(mse, 2)}")
print(f"R-squared: {round(r2, 2)}")

# #### Comments #####
# Overall, the results indicate that the random forest regression model, as
# currently configured, is not performing well for the task of predicting stock prices.
# The negative R-squared suggests that the model is not a good fit for the data, and
# both the MAE and MSE are relatively high, indicating a lack of accuracy in the predictions.



### Third training with logistic regression

# Define the features and binary target variable
features = ["High", "Low", "Volume", "Dividends"]

# Create a binary target variable based on criteria
df["stock_going_down"] = df["Close"] <= 5

# Split the dataset into training and testing sets
X = df[features]
y = df["stock_going_down"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

##### Comments #####
# The analysis of these results indicates that the model is performing very well in terms
# of predicting "False" cases (stocks not going down to 0). However, it completely fails to
# predict "True" cases (stocks going down to 0) because it labels all instances as "False".
# This suggests a severa class imbalance issue in the dataset, where the majority of stocks
# are "False", and the model does not learn to predict the minority class.