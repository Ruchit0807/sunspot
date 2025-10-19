import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="Sunspot Predictor", layout="wide")
st.title("🌞 Solar Activity - Sunspot Predictor")

# Create dataset (1700–2024)
years = np.arange(1700, 2025)
np.random.seed(42)
sunspots = np.abs(50 * np.sin((years - 1700) * np.pi / 5.5) + np.random.normal(0, 10, len(years)))
df = pd.DataFrame({'Year': years, 'Sunspot': sunspots})

# Train Polynomial Regression Model
X = df[['Year']]
y = df['Sunspot']
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# User input
user_year = st.number_input("Enter the year to predict sunspot number:", min_value=1980, max_value=2100, value=2030)

# Predict or fetch actual
if user_year in df['Year'].values:
    result = df.loc[df['Year'] == user_year, 'Sunspot'].values[0]
    st.success(f"The actual average sunspot number in {user_year} was: {result:.2f}")
else:
    prediction = model.predict(poly.transform([[user_year]]))[0]
    st.success(f"Predicted average sunspot number in {user_year}: {prediction:.2f}")

# Plot graph (1980 → user_year)
import matplotlib
matplotlib.use("Agg")  # Streamlit compatibility

fig, ax = plt.subplots(figsize=(10, 6))
subset = df[df['Year'] >= 1980]
ax.scatter(subset['Year'], subset['Sunspot'], color='blue', label='Actual Data')

future_years = np.arange(1980, user_year + 1)
future_X_poly = poly.transform(future_years.reshape(-1, 1))
predicted_sunspots = model.predict(future_X_poly)
ax.plot(future_years, predicted_sunspots, color='red', linestyle='--', label='Model Trend')

ax.scatter(user_year, model.predict(poly.transform([[user_year]])), color='green', s=100, label=f'Year {user_year}')
ax.set_title('Solar Activity (Sunspot Number Prediction)')
ax.set_xlabel('Year')
ax.set_ylabel('Average Sunspot Number')
ax.legend()
ax.grid(True)

st.pyplot(fig)