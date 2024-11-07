
import numpy as np
import pandas as pd
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
#iris = load_iris()
#iris.shape
iris=pd.read_csv("g:\\python\iris.csv")
X=iris.drop('variety', axis=1)
y=iris["variety"]   

#feature_names=['sepal.length','sepal.width','petal.length','petal.width']
#target_names=['variety']
##X = iris.data  # Features: sepal length, sepal width, petal length, petal width
#y = iris.target  # Target: species of iris
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plt.figure(figsize=(12,8))

plot_tree(clf, filled=True, feature_names=X.columns, class_names=y)
plt.title("Decision Tree for Iris Dataset")

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
plt.show()
