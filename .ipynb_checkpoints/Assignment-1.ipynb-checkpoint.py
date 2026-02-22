# =====================================================
# Assignment-1 : Linear Regression (Attrition Dataset)
# =====================================================

# 1️⃣ Import Libraries
import pandas as pd
import sklearn
print(sklearn.__version__)

# Load dataset
df = pd.read_csv(r"C:\Users\vikram sahani\Desktop\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert target column first
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Define X and y AFTER mapping
X = df[['Age', 'MonthlyIncome', 'JobLevel',
        'DistanceFromHome', 'Education',
        'JobSatisfaction', 'TotalWorkingYears',
        'YearsAtCompany']]   # total 8 features

y = df['Attrition']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred)) 

# ============================================
# Linear Regression (Single Feature)
# Straight Line + RSS Calculation
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1️⃣ Load Dataset
df = pd.read_csv(r"C:\Users\Kshama\OneDrive\Desktop\Assignment-01 ML\data\dataset")

# 2️⃣ Convert Attrition to numeric
df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})

# 3️⃣ Take Only One Feature for Straight Line
X = df[["MonthlyIncome"]]   # Single independent variable
y = df["Attrition"]         # Dependent variable

# 4️⃣ Create Model
model = LinearRegression()
model.fit(X, y)

# 5️⃣ Predictions
y_pred = model.predict(X)

# 6️⃣ Calculate RSS (Residual Sum of Squares)
rss = np.sum((y - y_pred) ** 2)
print("Residual Sum of Squares (RSS):", rss)

# 7️⃣ Plot Straight Line
plt.figure(figsize=(6,5))
plt.scatter(X, y)                # Actual points
plt.plot(X, y_pred)              # Regression straight line
plt.xlabel("Monthly Income")
plt.ylabel("Attrition")
plt.title("Linear Regression - Straight Line")
plt.show()