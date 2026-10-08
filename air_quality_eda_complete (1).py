# ============================================================
# EDA COURSE-END PROJECT
# Exploratory Analysis of Air Quality and Pollution Patterns
# Across Indian Cities
# Dataset: Kaggle - Air Quality Data in India
# Primary file: city_day.csv
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# 2. LOAD DATASET
# Upload city_day.csv to Google Colab, then run:
df = pd.read_csv("/content/city_day.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
display(df.head())

# 3. BASIC DATA EXPLORATION
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
display(df.describe(include="all").T)

# 4. MISSING VALUES
missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values("Missing Count", ascending=False)

print("\nMissing-value summary:")
display(missing)

# 5. DUPLICATES
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().copy()
print("Shape after removing exact duplicates:", df.shape)

# 6. DATE PREPARATION
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("\nInvalid dates:", df["Date"].isna().sum())

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()
df["Month_Number"] = df["Date"].dt.month

# 7. NUMERIC CONVERSION
pollutant_cols = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

available_pollutants = [c for c in pollutant_cols if c in df.columns]

for col in available_pollutants:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "AQI" in df.columns:
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

# 8. HANDLE MISSING POLLUTANT VALUES
# City-wise median, followed by overall median for any remainder.
for col in available_pollutants:
    df[col] = df.groupby("City")[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())

# Keep AQI missing values as missing because AQI is a primary
# analysis variable; do not invent AQI values.
aqi_df = df.dropna(subset=["AQI"]).copy()

# 9. DATA SUMMARY AFTER CLEANING
print("\n========== CLEANED DATA SUMMARY ==========")
print("Records:", len(df))
print("Columns:", len(df.columns))
print("Cities:", df["City"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("AQI records:", len(aqi_df))

# ============================================================
# ANALYTICAL QUESTION 1
# How does average AQI change over the years?
# Visualization: Line chart
# ============================================================
yearly_aqi = (
    aqi_df.groupby("Year")["AQI"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly_aqi, x="Year", y="AQI", marker="o")
plt.title("Average AQI Trend Over the Years")
plt.xlabel("Year")
plt.ylabel("Average AQI")
plt.xticks(yearly_aqi["Year"])
plt.tight_layout()
plt.show()

# ============================================================
# ANALYTICAL QUESTION 2
# Which cities have the highest and lowest average AQI?
# Visualization: Bar chart
# ============================================================
city_aqi = (
    aqi_df.groupby("City")["AQI"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

plt.figure(figsize=(12, 7))
sns.barplot(x=city_aqi["mean"], y=city_aqi.index)
plt.title("Average AQI by City")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.tight_layout()
plt.show()

print("\nTop 10 cities by average AQI:")
display(city_aqi.head(10))

print("\nBottom 10 cities by average AQI:")
display(city_aqi.tail(10).sort_values("mean"))

# ============================================================
# ANALYTICAL QUESTION 3
# Which pollutants have the strongest relationship with AQI?
# Visualization: Correlation heatmap
# ============================================================
corr_cols = [c for c in available_pollutants + ["AQI"] if c in df.columns]
corr_matrix = df[corr_cols].corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Between AQI and Pollutants")
plt.tight_layout()
plt.show()

print("\nCorrelation with AQI:")
display(
    corr_matrix["AQI"]
    .drop("AQI")
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .to_frame("Correlation with AQI")
)

# ============================================================
# ANALYTICAL QUESTION 4
# How does average PM2.5 vary across months?
# Visualization: Line chart
# ============================================================
if "PM2.5" in df.columns:
    monthly_pm25 = (
        df.groupby("Month_Number")["PM2.5"]
        .mean()
        .reset_index()
    )

    monthly_pm25["Month_Name"] = pd.to_datetime(
        monthly_pm25["Month_Number"], format="%m"
    ).dt.month_name().str[:3]

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=monthly_pm25,
        x="Month_Name",
        y="PM2.5",
        marker="o"
    )
    plt.title("Monthly Average PM2.5")
    plt.xlabel("Month")
    plt.ylabel("Average PM2.5")
    plt.tight_layout()
    plt.show()

# ============================================================
# ANALYTICAL QUESTION 5
# How does AQI vary across seasons?
# Visualization: Box plot
# ============================================================
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

aqi_df["Season"] = aqi_df["Month"].apply(assign_season)

season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

plt.figure(figsize=(10, 5))
sns.boxplot(
    data=aqi_df,
    x="Season",
    y="AQI",
    order=season_order
)
plt.title("AQI Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

# ============================================================
# ANALYTICAL QUESTION 6
# What is the distribution of AQI values?
# Visualization: Histogram
# ============================================================
plt.figure(figsize=(10, 5))
sns.histplot(data=aqi_df, x="AQI", bins=30, kde=True)
plt.title("Distribution of AQI Values")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# ANALYTICAL QUESTION 7
# What is the relationship between PM2.5 and AQI?
# Visualization: Scatter plot
# ============================================================
if "PM2.5" in aqi_df.columns:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=aqi_df,
        x="PM2.5",
        y="AQI",
        alpha=0.4
    )
    plt.title("Relationship Between PM2.5 and AQI")
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()

# ============================================================
# ANALYTICAL QUESTION 8
# What is the distribution of AQI categories?
# Visualization: Bar chart
# ============================================================
if "AQI_Bucket" in aqi_df.columns:
    plt.figure(figsize=(10, 5))
    order = aqi_df["AQI_Bucket"].value_counts().index

    sns.countplot(
        data=aqi_df,
        x="AQI_Bucket",
        order=order
    )

    plt.title("Distribution of AQI Categories")
    plt.xlabel("AQI Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    print("\nAQI category counts:")
    display(aqi_df["AQI_Bucket"].value_counts())

# ============================================================
# OPTIONAL VISUALIZATION 9
# Top 10 cities by average PM2.5
# ============================================================
if "PM2.5" in df.columns:
    top_pm25 = (
        df.groupby("City")["PM2.5"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_pm25.values, y=top_pm25.index)
    plt.title("Top 10 Cities by Average PM2.5")
    plt.xlabel("Average PM2.5")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()

# ============================================================
# FINAL PROJECT STATISTICS
# ============================================================
print("\n========== FINAL PROJECT STATISTICS ==========")

print("Number of records:", len(df))
print("Number of columns:", len(df.columns))
print("Number of cities:", df["City"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())

print("\nOverall average AQI:", round(aqi_df["AQI"].mean(), 2))
print("Highest observed AQI:", aqi_df["AQI"].max())
print("Lowest observed AQI:", aqi_df["AQI"].min())

print("\nCity with highest average AQI:")
print(city_aqi.index[0], "->", round(city_aqi.iloc[0]["mean"], 2))

print("\nCity with lowest average AQI:")
print(city_aqi.index[-1], "->", round(city_aqi.iloc[-1]["mean"], 2))

print("\nMissing values after cleaning:")
display(df.isnull().sum().sort_values(ascending=False).head(20))

# ============================================================
# END
# ============================================================
print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
