# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load house.csv, sanitize column names, and extract features (Size, Bedrooms) and targets (Price, Occupants).
2. Use StandardScaler to transform features into a standard scale (mean=0, variance=1), ensuring consistent model training.
3. Set up two SGDRegressor models with a constant learning rate of 0.01 and a maximum of 1,000 iterations.
4. Perform gradient descent on the scaled features to map inputs to the two specific output targets (Price and Occupants).Take user input, apply the same scaler used during training, and output the model predictions (rounding the occupant count).

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sridharan B
RegisterNumber:  212225230272
*/
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("house.csv")
data.columns = data.columns.str.strip()
X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = scaler.transform([[size, bed]])
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)
print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:

<img width="1210" height="177" alt="image" src="https://github.com/user-attachments/assets/4fb5f529-c83a-4504-a526-f64e6024c669" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
