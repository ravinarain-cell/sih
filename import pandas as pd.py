import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

np.random.seed(42)
num_records = 500

data = {
    "day_of_week": np.random.randint(1, 6, num_records),
    "time_of_day": np.random.randint(8, 17, num_records),
    "weather_condition": np.random.randint(1, 4, num_records),
    "enrolled_students": np.random.randint(30, 100, num_records),
}

df = pd.DataFrame(data)

actual_attendance = df["enrolled_students"].astype(float)
actual_attendance -= np.where(df["day_of_week"] == 5, df["enrolled_students"] * 0.15, 0)
actual_attendance -= np.where(df["time_of_day"] == 8, df["enrolled_students"] * 0.10, 0)
actual_attendance -= np.where(df["weather_condition"] == 3, df["enrolled_students"] * 0.20, 0)
actual_attendance += np.random.normal(0, 2, num_records)

df["actual_attendance"] = np.clip(actual_attendance, 0, df["enrolled_students"]).round().astype(int)

X = df[["day_of_week", "time_of_day", "weather_condition", "enrolled_students"]]
y = df["actual_attendance"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "attendance_model.joblib")
print("Model saved successfully as attendance_model.joblib")
