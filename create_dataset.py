import pandas as pd
import numpy as np

# Create same random data every time
np.random.seed(42)

# Number of records
n = 1000

# Create data
temperature = np.random.randint(18, 36, n)
occupants = np.random.randint(0, 6, n)
lights_on = np.random.randint(0, 11, n)
fans_on = np.random.randint(0, 6, n)

ac_hours = np.round(
    np.random.uniform(0, 10, n), 1
)

appliance_hours = np.round(
    np.random.uniform(0, 10, n), 1
)

# Calculate power consumption
power_consumption = (
    lights_on * 0.1
    + fans_on * 0.07
    + ac_hours * 1.2
    + appliance_hours * 0.5
    + np.random.uniform(0, 2, n)
)

# Create wastage label
wastage = (
    (power_consumption > 8)
    | ((occupants <= 1) & (power_consumption > 5))
    | ((ac_hours > 6) & (occupants <= 2))
).astype(int)

# Create table
data = pd.DataFrame({
    "temperature": temperature,
    "occupants": occupants,
    "lights_on": lights_on,
    "fans_on": fans_on,
    "ac_hours": ac_hours,
    "appliance_hours": appliance_hours,
    "power_consumption": np.round(
        power_consumption, 2
    ),
    "wastage": wastage
})

# Save dataset
data.to_csv(
    "electricity_data.csv",
    index=False
)

print("Dataset created successfully!")
print()
print(data.head())
print()
print("Number of records:", len(data))