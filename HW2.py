import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('weather.csv')
if df.isnull().sum().any():
    df.fillna(df.mean(), inplace=True)
if 'RainTomorrow' in df.columns:
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model_1 = MLPClassifier(hidden_layer_sizes=(1,), solver="adam", max_iter=10, learning_rate_init=0.001, random_state=42)
model_1.fit(X_train, y_train)
model_2 = MLPClassifier(hidden_layer_sizes=(1,), solver="adam", max_iter=3, learning_rate_init=0.0001, random_state=42)
model_2.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)
y_pred_2 = model_2.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred_1)
accuracy_2 = accuracy_score(y_test, y_pred_2)
print("Accuracy of Model 1 (max_iter=10, learning_rate_init=0.001):", accuracy_1)
print("Accuracy of Model 2 (max_iter=3, learning_rate_init=0.0001):", accuracy_2)
if accuracy_1 > accuracy_2:
    print("Model 1 performs better.")
elif accuracy_1 < accuracy_2:
    print("Model 2 performs better.")
else:
    print("Both models perform equally well.")
