from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

path = 'cardio-dataset.csv'
data = pd.read_csv(path)

print("Original dataset")
print(data.head(5), '\n')

data = pd.concat([data, pd.get_dummies(data.SEX, prefix='SEX')], axis=1)
data.drop('SEX', axis=1, inplace=True)

data = pd.concat([data, pd.get_dummies(data.SMOKE_, prefix='SMOKER')], axis=1)
data.drop('SMOKE_', axis=1, inplace=True)

data = pd.concat([data, pd.get_dummies(data.BPMED, prefix='BPMEDS')], axis=1)
data.drop('BPMED', axis=1, inplace=True)

data = pd.concat([data, pd.get_dummies(data.DIAB_01, prefix='DIABETIC')], axis=1)
data.drop('DIAB_01', axis=1, inplace=True)

data[['TC','HDL']] = MinMaxScaler().fit_transform(data[['TC','HDL']])

data.rename({'SEX_1':'Male',
             'SEX_2':'Female', 
             'SMOKER_0':'Smoker', 
             'SMOKER_1':'Non-Smoker', 
             'BPMEDS_1':'Has BP issues',
             'BPMEDS_2':'No BP issues',
             'DIABETIC_0':'Is diabetic',
             'DIABETIC_1':'Is not diabetic'},axis='columns', inplace=True)

print("Processed Datset")
print(data.head(5),'\n')

X = data.drop('RISK', axis=1)
y = data.RISK

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R^2 score: ", round(model.score(X_test,y_test), 3), '\n')

print("Mean Squared Error: ", round(metrics.mean_squared_error(y_test,y_pred), 3), '%\n')

result = pd.DataFrame(list(zip(y_test,y_pred)),columns=['Test values','Predicted values'])

print('Comparision')
print(result)