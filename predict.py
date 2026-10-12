import joblib
import pandas as pd

# Load trained model
model = joblib.load("electricity_wastage_model.pkl")

print("====================================")
print(" SMART ELECTRICITY WASTAGE DETECTOR ")
print("====================================")

# Take input from user
temperature = float(input("Enter temperature (°C): "))
occupants = int(input("Enter number of occupants: "))
lights_on = int(input("Enter number of lights ON: "))
fans_on = int(input("Enter number of fans ON: "))
ac_hours = float(input("Enter AC usage hours: "))
appliance_hours = float(input("Enter appliance usage hours: "))
power_consumption = float(input("Enter power consumption (kWh): "))

# Create input data
input_data = pd.DataFrame([[
    temperature,
    occupants,
    lights_on,
    fans_on,
    ac_hours,
    appliance_hours,
    power_consumption
]], columns=[
    "temperature",
    "occupants",
    "lights_on",
    "fans_on",
    "ac_hours",
    "appliance_hours",
    "power_consumption"
])

# Make prediction
prediction = model.predict(input_data)

# Display result
if prediction[0] == 1:
    print("\nELECTRICITY WASTAGE DETECTED!")
    print("Please reduce unnecessary electricity usage.")
else:
    print("\nNORMAL ELECTRICITY USAGE")
    print("Electricity consumption is within the normal range.")