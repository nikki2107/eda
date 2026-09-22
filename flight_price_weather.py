"""
Review 2 - Weather-Aware Demo
Flight Price Prediction + Delay Risk, adjusted using LIVE weather data

Requires:
  - flight_price_cleaned.csv (same folder, from flight_price_eda.py)
  - A free OpenWeatherMap API key: https://openweathermap.org/api
    (sign up, verify email, copy key from API keys tab, wait ~10 min to activate)
  - pip3 install requests scikit-learn pandas numpy

Run: python3 flight_price_weather.py

NOTE ON HONESTY FOR YOUR REVIEW:
The flight price dataset has NO delay or weather columns. So this script
combines two different things, and you should say this out loud in your demo:
  1. A trained ML model (Random Forest) predicting PRICE from real data.
  2. A transparent RULE-BASED heuristic estimating DELAY RISK from live
     weather severity, since no historical delay-vs-weather dataset exists
     here. This is disclosed, not hidden - it's a reasonable simulation
     layer on top of the real model, and a good "future work" discussion
     point (e.g. "with a delay-labeled dataset, this heuristic could be
     replaced by a second trained classifier").
"""

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
OPENWEATHER_API_KEY = input("Enter your OpenWeatherMap API key: ").strip()
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# Maps dataset city names -> real-world city names for the weather API
# (edit this if your dataset uses different city names)
CITY_MAP = {
    "Delhi": "Delhi,IN",
    "Mumbai": "Mumbai,IN",
    "Bangalore": "Bangalore,IN",
    "Kolkata": "Kolkata,IN",
    "Hyderabad": "Hyderabad,IN",
    "Chennai": "Chennai,IN",
}

# ---------------------------------------------------------
# 2. LOAD DATA + TRAIN PRICE MODEL
# ---------------------------------------------------------
print("\nLoading data and training price model, please wait...\n")

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
# 3. HELPERS: menu selection
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
# 4. LIVE WEATHER FETCH
# ---------------------------------------------------------
def get_weather(city_name):
    """
    Calls OpenWeatherMap current weather endpoint for the given city.
    Returns a dict with condition, wind speed (m/s), and visibility (m).
    Returns None if the request fails.
    """
    query_city = CITY_MAP.get(city_name, city_name)
    params = {
        "q": query_city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "visibility": data.get("visibility", 10000),
            "temp": data["main"]["temp"],
        }
    except Exception as e:
        print(f"Weather lookup failed ({e}). Using neutral/clear defaults.")
        return {
            "condition": "Clear",
            "description": "unavailable - defaulted to clear",
            "wind_speed": 3.0,
            "visibility": 10000,
            "temp": 25.0,
        }


# ---------------------------------------------------------
# 5. WEATHER SEVERITY SCORING (rule-based, transparent)
# ---------------------------------------------------------
CONDITION_SEVERITY = {
    "Thunderstorm": 0.9,
    "Snow": 0.8,
    "Tornado": 1.0,
    "Squall": 0.85,
    "Rain": 0.5,
    "Drizzle": 0.3,
    "Mist": 0.3,
    "Fog": 0.5,
    "Haze": 0.25,
    "Smoke": 0.3,
    "Dust": 0.35,
    "Sand": 0.35,
    "Ash": 0.6,
    "Clouds": 0.1,
    "Clear": 0.0,
}


def weather_severity(weather):
    base = CONDITION_SEVERITY.get(weather["condition"], 0.2)

    wind_penalty = min(weather["wind_speed"] / 25, 1.0) * 0.3
    visibility_penalty = max(0, (10000 - weather["visibility"]) / 10000) * 0.3

    severity = base + wind_penalty + visibility_penalty
    return round(min(severity, 1.0), 3)


# ---------------------------------------------------------
# 6. PRICE PREDICTION (ML model)
# ---------------------------------------------------------
def predict_price(user_row):
    encoded = user_row.copy()
    for col in categorical_cols:
        encoded[col] = encoders[col].transform([str(encoded[col])])[0]
    row_df = pd.DataFrame([encoded])[feature_cols]

    tree_preds = np.array([tree.predict(row_df)[0] for tree in model.estimators_])
    return tree_preds.mean(), tree_preds.std()


# ---------------------------------------------------------
# 7. WEATHER-ADJUSTED OUTPUTS (heuristic layer)
# ---------------------------------------------------------
def weather_adjusted_price(base_price, severity):
    """
    Heuristic: severe weather increases rebooking/demand volatility,
    so we apply a small surcharge proportional to severity.
    Disclosed assumption, tune freely for your report.
    """
    surcharge_pct = severity * 0.08  # up to 8% at max severity
    adjusted = base_price * (1 + surcharge_pct)
    return adjusted, surcharge_pct * 100


def delay_risk_probability(severity, stops):
    """
    Heuristic: base delay risk + weather severity + stop count penalty.
    Base rate loosely reflects typical on-time performance industry figures.
    """
    base_rate = 0.15
    stop_penalty = {"zero": 0.0, "one": 0.08, "two_or_more": 0.15}
    penalty = stop_penalty.get(str(stops).lower().replace(" ", "_"), 0.05)

    risk = base_rate + severity * 0.5 + penalty
    return round(min(risk, 0.95), 3)


# ---------------------------------------------------------
# 8. INTERACTIVE SESSION
# ---------------------------------------------------------
def run_session():
    print("=" * 55)
    print("WEATHER-AWARE FLIGHT PRICE + DELAY RISK PREDICTOR")
    print("=" * 55)

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

    print(f"\nFetching live weather for {source}...")
    weather = get_weather(source)
    severity = weather_severity(weather)

    base_price, price_std = predict_price(user_row)
    adjusted_price, surcharge_pct = weather_adjusted_price(base_price, severity)
    delay_risk = delay_risk_probability(severity, stops)

    print("\n" + "-" * 55)
    print("LIVE WEATHER SNAPSHOT")
    print("-" * 55)
    print(f"City: {source}")
    print(f"Condition: {weather['condition']} ({weather['description']})")
    print(f"Temp: {weather['temp']} C | Wind: {weather['wind_speed']} m/s | Visibility: {weather['visibility']} m")
    print(f"Weather severity score: {severity} (0 = clear, 1 = severe)")

    print("\n" + "-" * 55)
    print("PRICE PREDICTION")
    print("-" * 55)
    print(f"Base ML-predicted price: {base_price:.0f}")
    print(f"Weather-adjusted price: {adjusted_price:.0f} (+{surcharge_pct:.1f}% surcharge)")
    print(f"Model uncertainty (std across trees): {price_std:.0f}")

    print("\n" + "-" * 55)
    print("DELAY RISK ESTIMATE")
    print("-" * 55)
    print(f"Estimated delay probability: {delay_risk * 100:.1f}%")
    if delay_risk >= 0.5:
        print("Risk level: HIGH - consider buffer time for connections.")
    elif delay_risk >= 0.3:
        print("Risk level: MODERATE")
    else:
        print("Risk level: LOW")
    print("-" * 55)


# ---------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        run_session()
        again = input("\nRun another prediction? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye.")
            break
