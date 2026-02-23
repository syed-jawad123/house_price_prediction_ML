# house_price_prediction_fixed.py
# House Price Prediction – Ready-to-run with sample dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------
# Step 1: Sample Dataset
# ------------------------
# Creating a small sample dataset with 10 rows
data = pd.DataFrame({
    'LotArea': [8450, 9600, 11250, 9550, 14260, 14115, 10084, 10382, 6120, 7420],
    'OverallQual': [7, 6, 7, 7, 8, 5, 8, 7, 7, 5],
    'YearBuilt': [2003, 1976, 2001, 1915, 2000, 1993, 2004, 1973, 1931, 1939],
    'TotalBsmtSF': [856, 1262, 920, 756, 1145, 796, 1686, 996, 1329, 756],
    'GrLivArea': [1710, 1262, 1786, 1717, 2198, 1362, 1694, 2090, 1774, 1077],
    'FullBath': [2, 2, 2, 1, 2, 2, 2, 1, 2, 1],
    'GarageCars': [2, 2, 2, 3, 3, 2, 3, 3, 2, 1],
    'SalePrice': [208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000, 129900, 118000]
})

print("Sample Dataset:")
print(data)

# ------------------------
# Step 2: Features & Target
# ------------------------
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# ------------------------
# Step 3: Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Step 4: Feature Scaling
# ------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------
# Step 5: Train Models
# ------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ------------------------
# Step 6: Evaluate Models
# ------------------------
def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest Regressor")

# ------------------------
# Step 7: Visualization
# ------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: Actual vs Predicted House Prices')
plt.show()