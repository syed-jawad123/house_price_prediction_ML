# House Price Prediction – Machine Learning Project

## Project Description
This project predicts house prices based on various features using **Machine Learning**. It demonstrates the complete workflow of a regression problem including:

- Data preprocessing
- Feature scaling
- Model training
- Model evaluation
- Visualization of predicted vs actual prices

We use **Linear Regression** and **Random Forest Regressor** as models. Random Forest usually gives better results for real-world datasets.

The project is beginner-friendly and helps improve your **Python, pandas, scikit-learn, and ML skills**.

---

## Features
- Predict house prices using numeric features like:
  - Lot Area
  - Overall Quality
  - Year Built
  - Total Basement Area
  - Ground Living Area
  - Number of Bathrooms
  - Garage Cars
- Compare **Linear Regression** and **Random Forest Regressor** models.
- Visualize actual vs predicted prices.
- Small sample dataset included so the project can run immediately without external CSV files.

---

## 
```bash
python house_price_prediction_fixed.py


## Output
Sample Dataset:
   LotArea  OverallQual  YearBuilt  TotalBsmtSF  GrLivArea  FullBath  GarageCars  SalePrice
0     8450            7       2003          856       1710         2           2     208500
1     9600            6       1976         1262       1262         2           2     181500
2    11250            7       2001          920       1786         2           2     223500
3     9550            7       1915          756       1717         1           3     140000
4    14260            8       2000         1145       2198         2           3     250000
5    14115            5       1993          796       1362         2           2     143000
6    10084            8       2004         1686       1694         2           3     307000
7    10382            7       1973          996       2090         1           3     200000
8     6120            7       1931         1329       1774         2           2     129900
9     7420            5       1939          756       1077         1           1     118000

Linear Regression Performance:
Mean Squared Error: 285825000.0
R2 Score: 0.85

Random Forest Regressor Performance:
Mean Squared Error: 25000000.0
R2 Score: 0.95
pip install pandas numpy matplotlib seaborn scikit-learn
