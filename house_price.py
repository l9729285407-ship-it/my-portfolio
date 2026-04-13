import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Dummy Data Taiyar Karna (Real project mein aap CSV file use karenge)
data = {
    'Area_sqft': [1500, 2000, 1200, 2500, 1800, 2200, 1100, 3000],
    'Bedrooms': [3, 4, 2, 4, 3, 4, 2, 5],
    'Price': [450000, 600000, 350000, 750000, 520000, 650000, 320000, 900000]
}
df = pd.DataFrame(data)

# 2. X (Features) aur y (Target) ko alag karna
X = df[['Area_sqft', 'Bedrooms']]
y = df['Price']

# 3. Data ko Train aur Test mein baantna (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Linear Regression Model banana aur train karna
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction karna
y_pred = model.predict(X_test)

# 6. Check karna ki model kitna sahi hai
print(f"Predicted Prices: {y_pred}")
print(f"Actual Prices: {y_test.values}")

# Ek naye ghar ki keemat check karein (2100 sqft, 3 Bedrooms)
new_house = np.array([[2100, 3]])
prediction = model.predict(new_house)
print(f"\nPrice for 2100 sqft house: ${prediction[0]:,.2f}")