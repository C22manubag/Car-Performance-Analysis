import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load built-in dataset
data = pd.read_csv("mtcars.csv")

# Train the model
X = data[["hp", "wt", "disp"]]
y = data["mpg"]
model = LinearRegression().fit(X, y)

st.title("🚗 Fuel Economy Predictor (MPG)")
st.markdown("Predict car fuel economy using horsepower, weight, and engine displacement.")

# Manual Prediction
st.header("🔧 Predict MPG Manually")

hp = st.number_input("Horsepower (hp)", min_value=50, max_value=400, value=110)
wt = st.number_input("Weight (wt in 1000 lbs)", min_value=1.0, max_value=6.0, value=3.0)
disp = st.number_input("Displacement (disp in cubic inches)", min_value=50.0, max_value=500.0, value=200.0)

if st.button("Predict MPG"):
    input_data = pd.DataFrame([[hp, wt, disp]], columns=["hp", "wt", "disp"])
    result = model.predict(input_data)[0]
    st.success(f"Predicted MPG: {result:.2f}")

# Batch Prediction
st.header("📁 Batch Prediction from Your CSV")
uploaded_file = st.file_uploader("Upload a CSV with columns: hp, wt, disp", type="csv")

user_df = None

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    try:
        user_X = user_df[["hp", "wt", "disp"]]
        predictions = model.predict(user_X)
        user_df["Predicted MPG"] = predictions
        st.dataframe(user_df)
        st.download_button("Download Results as CSV", user_df.to_csv(index=False), file_name="predicted_mpg.csv")
    except Exception as e:
        st.error(f"Error: {e}")

# Dynamic Plot Section
st.header("📊 Model Performance")

if user_df is not None and "Predicted MPG" in user_df:
    # Check if actual MPG is available in uploaded data
    if "mpg" in user_df:
        plot_df = pd.DataFrame({
            "Actual MPG": user_df["mpg"],
            "Predicted MPG": user_df["Predicted MPG"]
        })
        plot_title = "Actual vs Predicted MPG (from uploaded data)"
    else:
        plot_df = pd.DataFrame({
            "Actual MPG": [None] * len(user_df),
            "Predicted MPG": user_df["Predicted MPG"]
        })
        plot_title = "Predicted MPG (uploaded data, no actual MPG to compare)"
else:
    predicted = model.predict(X)
    plot_df = pd.DataFrame({
        "Actual MPG": y,
        "Predicted MPG": predicted
    })
    plot_title = "Actual vs Predicted MPG (on mtcars dataset)"

# Plot
fig, ax = plt.subplots()

if plot_df["Actual MPG"].isnull().any():
    ax.plot(plot_df["Predicted MPG"], marker="o", label="Predicted MPG")
    ax.set_ylabel("Predicted MPG")
    ax.set_xlabel("Car Index")
else:
    ax.scatter(plot_df["Actual MPG"], plot_df["Predicted MPG"], color="blue", label="Predictions")
    ax.plot([plot_df["Actual MPG"].min(), plot_df["Actual MPG"].max()],
            [plot_df["Actual MPG"].min(), plot_df["Actual MPG"].max()],
            color="red", linestyle="--", label="Ideal")

    ax.set_xlabel("Actual MPG")
    ax.set_ylabel("Predicted MPG")

ax.set_title(plot_title)
ax.legend()
st.pyplot(fig)
