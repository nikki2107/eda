"""
Exploratory Data Analysis - Flight Price Prediction Dataset
Dataset: Kaggle "Flight Price Prediction" by shubhambathwal
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Before running: download the CSV from Kaggle and place it in the same
folder as this script, named 'flight_price.csv'.

Expected columns: Airline, Flight, Source City, Departure Time, Stops,
Arrival Time, Destination City, Class, Duration, Days_left, Price
(there is often also an unnamed index column - handled below)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("flight_price.csv")

# Drop stray index column if present (common in this dataset's export)
if df.columns[0].lower() in ["unnamed: 0", "index"]:
    df = df.drop(columns=[df.columns[0]])

# Standardize column names (this dataset ships with lowercase/underscore names)
df = df.rename(columns={
    "airline": "Airline",
    "flight": "Flight",
    "source_city": "Source City",
    "departure_time": "Departure Time",
    "stops": "Stops",
    "arrival_time": "Arrival Time",
    "destination_city": "Destination City",
    "class": "Class",
    "duration": "Duration",
    "days_left": "Days_left",
    "price": "Price",
})

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# ---------------------------------------------------------
# 2. BASIC METADATA / STRUCTURE
# ---------------------------------------------------------
print("\n--- Summary statistics (numeric) ---")
print(df.describe())

print("\n--- Summary statistics (categorical) ---")
print(df.describe(include="object"))

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

print("\n--- Unique values per categorical column ---")
cat_cols_check = df.select_dtypes(include="object").columns
for col in cat_cols_check:
    print(f"{col}: {df[col].nunique()} unique -> {df[col].unique()[:8]}")

# ---------------------------------------------------------
# 3. CLEANING
# ---------------------------------------------------------

# 3a. Drop exact duplicates if any
df = df.drop_duplicates()

# 3b. Handle missing values
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# 3c. Standardize text categories (strip whitespace, consistent case)
for col in cat_cols:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# 3d. Outlier check using IQR on key numeric columns
def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

print("\n--- Outlier counts (IQR method) ---")
for col in num_cols:
    print(f"{col}: {iqr_outlier_count(df[col])} outliers")

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------
# Price per unit duration -> normalizes cost by flight length
if "Duration" in df.columns:
    df["Price_per_Hour"] = df["Price"] / df["Duration"].replace(0, np.nan)

# Booking urgency bucket -> classic dynamic pricing signal
if "Days_left" in df.columns:
    df["Booking_Window"] = pd.cut(
        df["Days_left"],
        bins=[-1, 1, 7, 15, 30, 100],
        labels=["Last Minute (0-1d)", "Short (2-7d)", "Medium (8-15d)",
                "Long (16-30d)", "Very Long (30d+)"]
    )

# ---------------------------------------------------------
# 5. UNIVARIATE ANALYSIS
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["Price"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Ticket Price")

sns.histplot(df["Duration"], kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Flight Duration")

sns.histplot(df["Days_left"], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Distribution of Days Left Before Departure")

sns.countplot(data=df, x="Class", ax=axes[1, 1])
axes[1, 1].set_title("Class Counts (Economy vs Business)")

plt.tight_layout()
plt.savefig("01_univariate_distributions.png", dpi=150)
plt.close()

# Categorical counts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="Airline", ax=axes[0],
              order=df["Airline"].value_counts().index)
axes[0].set_title("Airline Counts")
axes[0].tick_params(axis="x", rotation=45)

sns.countplot(data=df, x="Stops", ax=axes[1])
axes[1].set_title("Number of Stops")

plt.tight_layout()
plt.savefig("02_categorical_counts.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 6. BIVARIATE ANALYSIS
# ---------------------------------------------------------

# Price vs days left -> the core "dynamic pricing" relationship
plt.figure()
sns.lineplot(data=df, x="Days_left", y="Price", hue="Class", errorbar=None)
plt.title("Average Price vs Days Left Before Departure")
plt.savefig("03_price_vs_days_left.png", dpi=150)
plt.close()

# Price by airline
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Airline", y="Price")
plt.title("Price Distribution by Airline")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("04_price_by_airline.png", dpi=150)
plt.close()

# Price by class
plt.figure()
sns.boxplot(data=df, x="Class", y="Price")
plt.title("Price by Class")
plt.savefig("05_price_by_class.png", dpi=150)
plt.close()

# Price by number of stops
plt.figure()
sns.boxplot(data=df, x="Stops", y="Price")
plt.title("Price by Number of Stops")
plt.savefig("06_price_by_stops.png", dpi=150)
plt.close()

# Price by booking window (urgency)
if "Booking_Window" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Booking_Window", y="Price")
    plt.title("Price by Booking Window (Urgency)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("07_price_by_booking_window.png", dpi=150)
    plt.close()

# ---------------------------------------------------------
# 7. CORRELATION HEATMAP
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("08_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 8. KEY TAKEAWAYS (printed for quick reference in your review)
# ---------------------------------------------------------
print("\n--- Correlation with Price ---")
print(corr["Price"].sort_values(ascending=False))

print("\n--- Average price by class ---")
print(df.groupby("Class")["Price"].mean())

print("\n--- Average price by booking window ---")
if "Booking_Window" in df.columns:
    print(df.groupby("Booking_Window", observed=True)["Price"].mean())

print("\nCleaned dataset shape:", df.shape)
df.to_csv("flight_price_cleaned.csv", index=False)
print("\nSaved cleaned dataset -> flight_price_cleaned.csv")
print("Saved 8 plots (01_ through 08_) in the current folder.")
