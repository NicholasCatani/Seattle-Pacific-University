### STEP 1: Load the Dataset (Okay)

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ClimateDataAnalysis").getOrCreate()

# Load the dataset
file_path = "C:\\Users\\Nicho\\Desktop\\GlobalLandTemperaturesByCountry.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Display the schema and some data
# print(df.printSchema())
# print(df.show(5))




### STEP 2: Data Preprocessing (Okay)

# Remove rows with null values
df_cleaned = df.dropna()

# Filter for data after 1900
df_filtered = df_cleaned.filter(df_cleaned["dt"] > "1900-01-01")
# print(df_filtered.show(15))




### STEP 3: Programming Tasks (Okay)

##############################
## Aggregate by Key
from pyspark.sql.functions import avg, max, min, year

# Group by country and aggregate
agg_df = df_filtered.groupby("Country").agg(
    avg("AverageTemperature").alias("AvgTemp"),
    max("AverageTemperature").alias("MaxTemp")
)
# print(agg_df.show(25))

##############################
## Window Functions
from pyspark.sql.window import Window

# Define window specification
window_spec = Window.partitionBy("Country").orderBy("dt").rowsBetween(-2, 2)

# Calculate moving average
df_with_moving_avg = df_filtered.withColumn("MovingAvgTemp", avg("AverageTemperature").over(window_spec))

# Select a few countries for detailed analysis
countries = ["Afghanistan", "United States", "India", "Australia"]
df_selected_countries = df_with_moving_avg.filter(df_with_moving_avg.Country.isin(countries))

# Display 7 samples for each selected country
# print(df_selected_countries.select("dt", "Country", "AverageTemperature", "MovingAvgTemp").show(30))

##############################
## Pivot Tables

# Extract year from date
df_yearly = df_filtered.withColumn("Year", year("dt"))

# Create pivot table
pivot_df = df_yearly.groupby("Year").pivot("Country").avg("AverageTemperature")
# print(pivot_df.select("Year", "Afghanistan", "United States", "India", "Australia").show(20))

##############################
## Multi-Level Aggregation

# Group by country and year, and aggregate
multi_level_agg_df = df_yearly.groupBy("Country", "Year").agg(
    avg("AverageTemperature").alias("AvgTemp"),
    min("AverageTemperature").alias("MinTemp"),
    max("AverageTemperature").alias("MaxTemp")
)

# Select relevant countries and display 10 records each
countries = ["Afghanistan", "Australia", "United States", "India"]
multi_level_agg_df_selected = multi_level_agg_df.filter(multi_level_agg_df.Country.isin(countries))

# print(multi_level_agg_df_selected.show(40))





