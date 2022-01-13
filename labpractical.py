# importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as scaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# reading dataset
df = pd.read_csv('winequality.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  # Splitting the column to be predicted

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  # 80:20 train test split


model = LinearRegression()  # Defining a Linear Regression model
model.fit(X_train, y_train) # Training the model on X_train

#y_pred = model.predict(X_test)
y_pred = np.round(model.predict(X_test)) # Predicting the values for X_test
y_pred[:50]

# Calculating model accuracy
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate", "\n")

# Calculating F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score: ',f1, "\n")

#Calculating R^2 score
print("R^2 Score: ", model.score(X_test,y_test), "\n")

# Calculating Mean Squared Error
print("Mean Squared Error: ", metrics.mean_squared_error(y_test,y_pred), "\n")

# Printing the test vs expected values
result = pd.DataFrame(list(zip(y_test,y_pred)),columns=['Test values','Predicted values'])
print("Expected vs Predicted values:")
print(result.head(10), "\n")

# Plotting graph of the above
x_ax = range(len(y_test))
y_pred = np.array(y_pred)
y_test = np.array(y_test)
plt.scatter(x_ax[:50], y_test[:50], label="original")
plt.scatter(x_ax[:50], y_pred[:50], label="predicted")
for i in range(50):
    plt.plot([x_ax[i],x_ax[i]],[y_pred[i],y_test[i]],c='black')
    
plt.title("test and predicted data (first 50 values)")
plt.xlabel('Patient')
plt.ylabel('Risk')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()