import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv("D:\\python\Heart_disease_statlog.csv")
y=df['target']
x=df.drop(['target'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)

# ایجاد مدل شبکه عصبی
def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(random.randint(10, 50), activation='relu'),
        layers.Dense(random.randint(5, 30), activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# آموزش و ارزیابی مدل
def train_and_evaluate_model(x_train, y_train, x_test, y_test):
    model = create_model(x_train.shape[1])  # ایجاد مدل
    model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1)  # آموزش مدل
    loss, accuracy = model.evaluate(x_test, y_test)  # ارزیابی مدل
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')  # نمایش نتایج

# آموزش و ارزیابی مدل
train_and_evaluate_model(x_train, y_train, x_test, y_test)