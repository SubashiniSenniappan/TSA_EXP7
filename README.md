# Ex.No: 07 AUTO REGRESSIVE MODEL
# Developed by Subashini S
# Reg no :212222240106
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import math

# Step 2: Load the dataset (weather data you uploaded)
file_path = '/content/weather_classification_data.csv'
data = pd.read_csv(file_path)

# Step 3: Add a synthetic 'date' column (assuming daily data)
data['date'] = pd.date_range(start='2022-01-01', periods=len(data), freq='D')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# For AutoRegressive analysis, let's use the 'Temperature' column as the target
data['target'] = data['Temperature']

# Step 4: Perform the Augmented Dickey-Fuller test for stationarity
result = adfuller(data['target'].dropna())
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
if result[1] > 0.05:
    print("The series is non-stationary. Differencing is required.")
else:
    print("The series is stationary. No differencing is needed.")

# Step 5: Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(data['target']) * 0.8)
train, test = data['target'][:train_size], data['target'][train_size:]

# Step 6: Fit the AutoRegressive model with 13 lags
model = AutoReg(train, lags=13)
model_fit = model.fit()

# Step 7: Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10,6))
plot_pacf(train, lags=13)
plt.show()

plt.figure(figsize=(10,6))
plot_acf(train, lags=13)
plt.show()

# Step 8: Make predictions on the test set
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Step 9: Calculate Mean Squared Error (MSE) for the test predictions
mse = mean_squared_error(test, predictions)
print(f"Test MSE: {mse:.4f}")
rmse = math.sqrt(mse)
print(f"Test RMSE: {rmse:.4f}")

# Step 10: Plot the original test data and predictions
plt.figure(figsize=(10,6))
plt.plot(test.index, test, label='Test Data', color='blue')
plt.plot(test.index, predictions, label='Predictions', color='orange')
plt.title('AutoRegressive Model Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Displaying test predictions and final predictions
print("\nTest Predictions:")
print(predictions)
```
### OUTPUT:
![Screenshot 2024-10-16 132841](https://github.com/user-attachments/assets/1b25c294-a409-4ced-909d-fabb6352f96c)
![image](https://github.com/user-attachments/assets/469de4fa-15a9-4d62-b663-2c7afa827e56)
![image](https://github.com/user-attachments/assets/caaec05a-7259-4eb0-bd23-123e5849bd04)
![Screenshot 2024-10-16 132941](https://github.com/user-attachments/assets/2ad6fa22-7ff0-45b0-a0e4-5b7dbb8ac8b8)
![image](https://github.com/user-attachments/assets/144c35de-abf8-4dd8-9af2-ad73e88c21db)
![image](https://github.com/user-attachments/assets/d5adcb02-2712-4025-8312-1ff8b389b34e)








### RESULT:
Thus we have successfully implemented the auto regression function using python.
