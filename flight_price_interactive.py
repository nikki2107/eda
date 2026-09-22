"""
Review 2 - Interactive Demo
Flight Price Prediction with Probability Estimate + Booking Recommendation

Requires flight_price_cleaned.csv (produced by flight_price_eda.py)
in the same folder.

Run: python3 flight_price_interactive.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. LOAD DATA + TRAIN MODEL (runs once at startup)
# ---------------------------------------------------------
print("Loading data and training model, please wait...\n")

df = pd.read_csv("flight_price_cleaned.csv")

categorical_cols = ["Airline", "Source City", "Departure Time", "Stops",
                     "Arrival Time", "Destination City", "Class"]
categorical_cols = [c for c in categorical_cols if c in df.columns]

model_df = df.copy()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

feature_cols = categorical_cols + ["Duration", "Days_left"]
feature_cols = [c for c in feature_cols if c in model_df.columns]

X = model_df[feature_cols]
y = model_df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

print("Model ready.\n")

# ---------------------------------------------------------
# 2. HELPER: pick from a numbered list (avoids typos)
# ---------------------------------------------------------
def choose_option(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


def choose_number(prompt, min_val=None, max_val=None, is_float=False):
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            val = float(raw) if is_float else int(raw)
            if min_val is not None and val < min_val:
                print(f"Must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Must be at most {max_val}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid number.")


# ---------------------------------------------------------
# 3. CORE: predict price + probability estimate
# ---------------------------------------------------------
def predict_with_probability(user_input_row):
    """
    Encodes the user's inputs, then asks EVERY tree in the forest
    for its own prediction. The spread across trees gives us:
      - a point estimate (mean)
      - a confidence range (based on std across trees)
      - a probability that the true price is below a threshold
    """
    encoded = user_input_row.copy()
    for col in categorical_cols:
        encoded[col] = encoders[col].transform([str(encoded[col])])[0]

    row_df = pd.DataFrame([encoded])[feature_cols]

    # Predictions from every individual tree in the forest
    tree_preds = np.array([tree.predict(row_df)[0] for tree in model.estimators_])

    mean_price = tree_preds.mean()
    std_price = tree_preds.std()
    low_90 = np.percentile(tree_preds, 5)
    high_90 = np.percentile(tree_preds, 95)

    return mean_price, std_price, low_90, high_90, tree_preds


def probability_price_drops_if_waiting(user_input_row, current_days_left):
    """
    Simulates the same flight at every days_left value BETWEEN NOW
    and departure, then computes what fraction of those future points
    have a lower predicted price than today. That fraction IS the
    probability estimate for 'should I wait?'
    """
    rows = []
    for days_left in range(1, current_days_left):
        row = user_input_row.copy()
        row["Days_left"] = days_left
        rows.append(row)

    if not rows:
        return None, None  # already at days_left = 1, nothing to wait for

    sim_df = pd.DataFrame(rows)
    for col in categorical_cols:
        sim_df[col] = encoders[col].transform(sim_df[col].astype(str))
    sim_df = sim_df[feature_cols]

    future_preds = model.predict(sim_df)
    current_price = predict_with_probability(user_input_row)[0]

    prob_drop = (future_preds < current_price).mean()
    best_day_index = np.argmin(future_preds)
    best_day = list(range(1, current_days_left))[best_day_index]
    best_price = future_preds[best_day_index]

    return prob_drop, (best_day, best_price)


# ---------------------------------------------------------
# 4. INTERACTIVE LOOP
# ---------------------------------------------------------
def run_session():
    print("=" * 50)
    print("FLIGHT PRICE PREDICTOR + BOOKING ADVISOR")
    print("=" * 50)

    airline = choose_option("Select airline:", sorted(df["Airline"].unique()))
    source = choose_option("Select source city:", sorted(df["Source City"].unique()))
    dest_options = sorted([c for c in df["Destination City"].unique() if c != source])
    dest = choose_option("Select destination city:", dest_options)
    dep_time = choose_option("Select departure time:", sorted(df["Departure Time"].unique()))
    arr_time = choose_option("Select arrival time:", sorted(df["Arrival Time"].unique()))
    stops = choose_option("Select number of stops:", sorted(df["Stops"].unique()))
    flight_class = choose_option("Select class:", sorted(df["Class"].unique()))
    duration = choose_number("Enter flight duration in hours (e.g. 2.5)", min_val=0.5, max_val=50, is_float=True)
    days_left = choose_number("Enter days left before departure", min_val=1, max_val=49)

    user_row = {
        "Airline": airline,
        "Source City": source,
        "Departure Time": dep_time,
        "Stops": stops,
        "Arrival Time": arr_time,
        "Destination City": dest,
        "Class": flight_class,
        "Duration": duration,
        "Days_left": days_left,
    }

    mean_price, std_price, low_90, high_90, tree_preds = predict_with_probability(user_row)

    print("\n" + "-" * 50)
    print("PREDICTION RESULT")
    print("-" * 50)
    print(f"Route: {source} -> {dest}, {airline}, {flight_class}")
    print(f"Predicted price: {mean_price:.0f}")
    print(f"90% confidence range: {low_90:.0f} to {high_90:.0f}")
    print(f"Model uncertainty (std across trees): {std_price:.0f}")

    prob_drop, best = probability_price_drops_if_waiting(user_row, days_left)

    print("\n" + "-" * 50)
    print("BOOKING RECOMMENDATION")
    print("-" * 50)
    if prob_drop is None:
        print("You're already at the last possible booking day - book now.")
    else:
        print(f"Probability price drops if you wait: {prob_drop * 100:.1f}%")
        best_day, best_price = best
        print(f"Cheapest predicted point: {best_day} days left (price ~ {best_price:.0f})")
        if prob_drop > 0.5:
            print("Recommendation: WAIT - price is likely to drop.")
        else:
            print("Recommendation: BOOK NOW - price is unlikely to get cheaper.")

    print("-" * 50)


# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        run_session()
        again = input("\nRun another prediction? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye.")
            break
