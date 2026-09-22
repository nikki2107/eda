"""
Review 2 - Implementation
Flight Price Prediction Model + "Best Time to Book" Recommender

Requires flight_price_cleaned.csv (produced by flight_price_eda.py)
in the same folder. Run flight_price_eda.py first if you haven't.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------
df = pd.read_csv("flight_price_cleaned.csv")
print("Loaded shape:", df.shape)

# ---------------------------------------------------------
# 2. PREPARE FEATURES
# ---------------------------------------------------------
model_df = df.copy()

categorical_cols = ["Airline", "Source City", "Departure Time", "Stops",
                     "Arrival Time", "Destination City", "Class"]
categorical_cols = [c for c in categorical_cols if c in model_df.columns]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

feature_cols = categorical_cols + ["Duration", "Days_left"]
feature_cols = [c for c in feature_cols if c in model_df.columns]

X = model_df[feature_cols]
y = model_df["Price"]

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. TRAIN MODEL
# ---------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. EVALUATE
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model performance ---")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# Feature importance
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)
print("\n--- Feature importance ---")
print(importance)

plt.figure(figsize=(8, 5))
sns.barplot(x=importance.values, y=importance.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("09_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 6. PRICE TRAJECTORY SIMULATION (the novel part)
# ---------------------------------------------------------
def simulate_trajectory(airline, source_city, dep_time, stops, arr_time,
                         dest_city, flight_class, duration):
    """
    Predict price for the SAME route/flight across every possible
    days_left value, to see how price changes as departure approaches.
    """
    rows = []
    for days_left in range(1, 50):
        row = {
            "Airline": airline,
            "Source City": source_city,
            "Departure Time": dep_time,
            "Stops": stops,
            "Arrival Time": arr_time,
            "Destination City": dest_city,
            "Class": flight_class,
            "Duration": duration,
            "Days_left": days_left,
        }
        rows.append(row)

    sim_df = pd.DataFrame(rows)

    # Encode using the SAME encoders fit during training
    for col in categorical_cols:
        sim_df[col] = encoders[col].transform(sim_df[col].astype(str))

    sim_df = sim_df[feature_cols]
    predicted_prices = model.predict(sim_df)

    result = pd.DataFrame({
        "Days_left": range(1, 50),
        "Predicted_Price": predicted_prices
    })
    return result


def recommend_booking_window(trajectory):
    """
    Given a price trajectory, recommend the days_left value
    with the lowest predicted price.
    """
    best_row = trajectory.loc[trajectory["Predicted_Price"].idxmin()]
    return int(best_row["Days_left"]), round(best_row["Predicted_Price"], 2)


# ---------------------------------------------------------
# 7. DEMO: run the recommender on one example route
# ---------------------------------------------------------
# NOTE: replace these values with actual categories present in your
# dataset (check df["Airline"].unique(), df["Source City"].unique(), etc.)
example_airline = df["Airline"].mode()[0]
example_source = df["Source City"].mode()[0]
example_dest = df["Destination City"].mode()[0]
example_dep_time = df["Departure Time"].mode()[0]
example_arr_time = df["Arrival Time"].mode()[0]
example_stops = df["Stops"].mode()[0]
example_class = df["Class"].mode()[0]
example_duration = df["Duration"].median()

trajectory = simulate_trajectory(
    airline=example_airline,
    source_city=example_source,
    dep_time=example_dep_time,
    stops=example_stops,
    arr_time=example_arr_time,
    dest_city=example_dest,
    flight_class=example_class,
    duration=example_duration,
)

best_day, best_price = recommend_booking_window(trajectory)

print(f"\n--- Demo recommendation ---")
print(f"Route: {example_source} -> {example_dest}, {example_airline}, {example_class}")
print(f"Best booking window: {best_day} days before departure")
print(f"Predicted price at that point: {best_price}")

# Plot the trajectory
plt.figure(figsize=(9, 5))
plt.plot(trajectory["Days_left"], trajectory["Predicted_Price"], marker="o", markersize=3)
plt.axvline(best_day, color="red", linestyle="--", label=f"Best: {best_day} days left")
plt.gca().invert_xaxis()  # so it reads left (far out) -> right (near departure)
plt.xlabel("Days left before departure")
plt.ylabel("Predicted price")
plt.title(f"Price Trajectory: {example_source} to {example_dest}")
plt.legend()
plt.tight_layout()
plt.savefig("10_price_trajectory_demo.png", dpi=150)
plt.close()

print("\nSaved 09_feature_importance.png and 10_price_trajectory_demo.png")
print("Model training + recommendation demo complete.")
