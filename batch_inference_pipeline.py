import os
import pandas as pd
import hopsworks
import joblib
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- CONFIG ----------------
PROJECT_NAME = "occupancy"
MODEL_NAME = "occupancy_rf"
MODEL_VERSION = 1 

# ---------------- SETUP -----------------
project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project=PROJECT_NAME,
    api_key_value="KBUFZFji5o45qBV7.pwW2LxTOPiDydHTElM4zNWpdb3ir6DrRSByPVIMsnkW9HCW8AOnrddOFilaBzHbW"
)
fs = project.get_feature_store()
mr = project.get_model_registry()

# ---------------- 1. GET DATA ----------------
weather_fg = fs.get_feature_group(name="skane_weather", version=1)
calendar_fg = fs.get_feature_group(name="sweden_calendar", version=1)

today = date.today()
tomorrow = today + timedelta(days=1)
weather_df = weather_fg.filter(weather_fg["event_time"] >= tomorrow).read()
calendar_df = calendar_fg.filter(calendar_fg["event_time"] >= today).read()
inference_df = pd.merge(weather_df, calendar_df, on=["date"], how="left")


# Select the exact features used in training (Order matters!)
feature_cols = [
    "hour", "month", "weekday", "is_weekend", 
    "is_holiday_se", "is_workday_se", 
    "temperature_2m", "precipitation", "windspeed_10m", "cloudcover"
]

X = inference_df[feature_cols]

# ---------------- 2. LOAD MODEL & PREDICT ----------------
print("Downloading and loading model...")
model_meta = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
model_dir = model_meta.download()
model = joblib.load(model_dir + "/model.pkl")

print("Predicting occupancy...")
preds = model.predict(X)

inference_df['predicted_occupancy'] = preds

# Map labels to numbers for plotting if they are strings (EMPTY, CROWDED...)
# Assuming order: EMPTY=0, MANY_SEATS=1, CROWDED=2 for visualization
label_map = {"EMPTY": 0, "MANY_SEATS_AVAILABLE": 1, "CROWDED": 2}
inference_df['occupancy_code'] = inference_df['predicted_occupancy'].map(label_map)

# ---------------- 3. VISUALIZE ----------------
print("Generating plot...")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Plotting
plt.plot(inference_df['event_time_x'], inference_df['occupancy_code'], marker='o', linestyle='-', color='#1EB182')

# Formatting
plt.yticks([0, 1, 2], ["Empty", "Seats Available", "Crowded"])
plt.title(f"Public Transport Occupancy Forecast (Next 5 Days)")
plt.xlabel("Date/Time")
plt.ylabel("Occupancy Level")
plt.xticks(rotation=45)
plt.tight_layout()

# Save image to the docs folder for GitHub Pages
output_dir = "docs/images"
os.makedirs(output_dir, exist_ok=True)
save_path = f"{output_dir}/occupancy_forecast.png"
plt.savefig(save_path)
print(f"Plot saved to {save_path}")