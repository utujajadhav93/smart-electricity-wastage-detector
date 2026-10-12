import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("electricity_data.csv")

print("======================================")
print("SMART ELECTRICITY WASTAGE DETECTOR")
print("======================================")
print()

print("Dataset loaded successfully!")
print("Total records:", len(data))
print()


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "temperature",
    "occupants",
    "lights_on",
    "fans_on",
    "ac_hours",
    "appliance_hours",
    "power_consumption"
]

X = data[features]
y = data["wastage"]

print("Features selected successfully!")
print("Features:", features)
print()


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print()


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training Random Forest model...")
print()


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("Model trained successfully!")
print()


# ==========================================
# 6. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("======================================")
print("MODEL PERFORMANCE")
print("======================================")

print("Accuracy:", round(accuracy * 100, 2), "%")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

model_file = "electricity_wastage_model.pkl"

joblib.dump(model, model_file)

print()
print("======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")
print("File created:", model_file)
print()
print("Training completed successfully!")