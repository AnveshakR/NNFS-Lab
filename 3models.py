import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris = load_iris() 
iris_data = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_data['class'] = iris.target
iris_data.head()
X = iris_data.values[:, 0:4] 
Y = iris_data.values[:,4] 
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)
algorithm_metrics = {}

# K Nearest Neighbor (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
predicted_targets = knn_model.predict(x_test)
algorithm_metrics["KNN (k=3)"] = {"ConfusionMatrix":confusion_matrix(y_test, predicted_targets), "ClassificationReport":classification_report(y_test, predicted_targets), "Accuracy": accuracy_score(predicted_targets,y_test)}
print("Classification Report:")
print(algorithm_metrics["KNN (k=3)"]["ClassificationReport"], end="\n")
print("ConfusionMatrix:")
print(algorithm_metrics["KNN (k=3)"]["ConfusionMatrix"], end="\n")
print("Accuracy")
print(algorithm_metrics["KNN (k=3)"]["Accuracy"]*100, "%", end="\n")

# Logistic Regression
regression_model = LogisticRegression()
regression_model.fit(x_train, y_train)
predicted_targets = regression_model.predict(x_test)
algorithm_metrics["Regression"] = {"ConfusionMatrix":confusion_matrix(y_test, predicted_targets), "ClassificationReport":classification_report(y_test, predicted_targets), "Accuracy": accuracy_score(predicted_targets,y_test)}
print("Classification Report:")
print(algorithm_metrics["Regression"]["ClassificationReport"], end="\n")
print("ConfusionMatrix:")
print(algorithm_metrics["Regression"]["ConfusionMatrix"], end="\n")
print("Accuracy")
print(algorithm_metrics["Regression"]["Accuracy"]*100, "%", end="\n")

# Naive Bayesian
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
predicted_targets = nb_model.predict(x_test)
algorithm_metrics["NaiveBayes"] = {"ConfusionMatrix":confusion_matrix(y_test, predicted_targets), "ClassificationReport":classification_report(y_test, predicted_targets), "Accuracy": accuracy_score(predicted_targets,y_test)}
print("Classification Report:")
print(algorithm_metrics["NaiveBayes"]["ClassificationReport"], end="\n")
print("ConfusionMatrix:")
print(algorithm_metrics["NaiveBayes"]["ConfusionMatrix"], end="\n")
print("Accuracy")
print(algorithm_metrics["NaiveBayes"]["Accuracy"]*100, "%", end="\n")

# Model Comparison in terms of accuracy:
print("Model - Accuracy")
for k,v in algorithm_metrics.items():
  print(k,"-",v["Accuracy"]*100, "%")