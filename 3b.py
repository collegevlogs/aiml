import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_csv('breast_cancer.csv')
df = df.iloc[:, :-1]
x = df.iloc[:, 2:].values
y = df['diagnosis'].values
df.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=500)
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
print("Training Accuracy:", accuracy_score(y_train, nb_model.predict(x_train)))
print("Testing Accuracy:", accuracy_score(y_test, nb_model.predict(x_test)))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, nb_model.predict(x_train)))
print("Testing Confusion Matrix:\n", confusion_matrix(y_test,nb_model.predict(x_test)))
print(classification_report(y_test, nb_model.predict(x_test)))
