import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("electricity_wastage_model.pkl")

# -------------------------------
# PAGE TITLE
# -------------------------------

st.title("⚡ Smart Electricity Wastage Detector")
st.write("Enter electricity usage details to detect wastage.")

# -------------------------------
# USER INPUTS
# -------------------------------

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=30.0
)

occupants = st.number_input(
    "Number of Occupants",
    min_value=1,
    max_value=20,
    value=4
)

lights_on = st.number_input(
    "Lights ON",
    min_value=0,
    max_value=50,
    value=5
)

fans_on = st.number_input(
    "Fans ON",
    min_value=0,
    max_value=20,
    value=3
)

ac_hours = st.number_input(
    "AC Usage (hours)",
    min_value=0.0,
    max_value=24.0,
    value=4.0
)

appliance_hours = st.number_input(
    "Appliance Usage (hours)",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

power_consumption = st.number_input(
    "Power Consumption (kWh)",
    min_value=0.0,
    max_value=100.0,
    value=8.0
)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("🔍 Detect Electricity Wastage"):

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

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    wastage_probability = probabilities[1] * 100
    no_wastage_probability = probabilities[0] * 100

    # ==================================================
    # FIRST OUTPUT - PREDICTION
    # ==================================================

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Electricity Wastage Detected!")
    else:
        st.success("✅ No Significant Electricity Wastage Detected.")

    # -------------------------------
    # WASTAGE PROBABILITY
    # -------------------------------

    st.metric(
        "Wastage Probability",
        f"{wastage_probability:.2f}%"
    )

    # -------------------------------
    # WASTAGE PROBABILITY GRAPH
    # -------------------------------

    st.subheader("📊 Wastage Probability Graph")

    probability_data = pd.DataFrame({
        "Status": ["No Wastage", "Wastage"],
        "Probability (%)": [
            no_wastage_probability,
            wastage_probability
        ]
    })

    st.bar_chart(
        probability_data.set_index("Status")
    )

    # ==================================================
    # 1. WASTAGE LEVEL
    # ==================================================

    st.subheader("🎯 Wastage Level")

    if wastage_probability <= 30:
        st.success("🟢 LOW WASTAGE")
        st.write("Electricity usage is under control.")

    elif wastage_probability <= 60:
        st.warning("🟡 MEDIUM WASTAGE")
        st.write("Some electricity usage can be reduced.")

    else:
        st.error("🔴 HIGH WASTAGE")
        st.write("High electricity wastage detected. Immediate saving actions are recommended.")

    # ==================================================
    # 2. ENERGY USAGE DASHBOARD
    # ==================================================

    st.subheader("⚡ Energy Usage Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Occupants", occupants)

    with col2:
        st.metric("💡 Lights ON", lights_on)

    with col3:
        st.metric("🌀 Fans ON", fans_on)

    with col4:
        st.metric("⚡ Power", f"{power_consumption} kWh")

    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric("❄️ AC Hours", ac_hours)

    with col6:
        st.metric("🔌 Appliance Hours", appliance_hours)

    with col7:
        st.metric("🌡️ Temperature", f"{temperature} °C")

    # ==================================================
    # 3. USAGE COMPARISON GRAPH
    # ==================================================

    st.subheader("📈 Usage Comparison Graph")

    usage_data = pd.DataFrame({
        "Appliance": [
            "Lights",
            "Fans",
            "AC",
            "Appliances"
        ],
        "Usage": [
            lights_on,
            fans_on,
            ac_hours,
            appliance_hours
        ]
    })

    st.bar_chart(
        usage_data.set_index("Appliance")
    )

    # ==================================================
    # 4. ENERGY SAVING SUGGESTIONS
    # ==================================================

    st.subheader("💡 Energy Saving Suggestions")

    if prediction == 1:

        st.warning("🔌 Turn off unused lights.")

        st.warning("❄️ Reduce AC usage when possible.")

        st.warning("⚡ Switch off appliances when they are not needed.")

        if lights_on > 5:
            st.info("💡 You have many lights ON. Consider switching off unnecessary lights.")

        if fans_on > 4:
            st.info("🌀 Consider switching off unused fans.")

        if ac_hours > 6:
            st.info("❄️ AC usage is relatively high. Try reducing AC operating hours.")

    else:

        st.success("🌱 Great Job!")

        st.success(
            "💚 Your electricity usage looks efficient. "
            "Keep saving energy and continue using appliances wisely."
        )

    # ==================================================
    # 5. INPUT DETAILS
    # ==================================================

    st.subheader("📋 Input Details")

    st.dataframe(input_data)