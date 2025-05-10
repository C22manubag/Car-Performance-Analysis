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

st.header("🔧 Predict MPG Manually")

hp = st.number_input("Horsepower (hp)", min_value=50, max_value=400, value=110)
wt = st.number_input("Weight (wt in 1000 lbs)", min_value=1.0, max_value=6.0, value=3.0)
disp = st.number_input("Displacement (disp in cubic inches)", min_value=50.0, max_value=500.0, value=200.0)

if st.button("Predict MPG"):
    result = model.predict([[hp, wt, disp]])[0]
    st.success(f"Predicted MPG: {result:.2f}")

# Upload your own CSV file
st.header("📁 Batch Prediction from Your CSV")
uploaded_file = st.file_uploader("Upload a CSV with columns: hp, wt, disp", type="csv")

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

# Show plot of actual vs predicted from built-in mtcars dataset
st.header("📊 Model Performance on mtcars Dataset")

predicted = model.predict(X)
comparison_df = pd.DataFrame({"Actual MPG": y, "Predicted MPG": predicted})

fig, ax = plt.subplots()
ax.scatter(comparison_df["Actual MPG"], comparison_df["Predicted MPG"], color="blue")
ax.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--")
ax.set_xlabel("Actual MPG")
ax.set_ylabel("Predicted MPG")
ax.set_title("Actual vs Predicted MPG (on mtcars dataset)")
st.pyplot(fig)
