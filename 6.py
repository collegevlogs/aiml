import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
data = pd.read_csv('pima-indians-diabetes.csv', delimiter=',')
X = data.iloc[:, :8]
y = data.iloc[:, 8]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=1,
                    validation_data=(x_test, y_test))
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.show()
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Training Accuracy: {train_acc*100:.2f}')
print(f'Test Accuracy: {test_acc*100:.2f}')
pred = model.predict(x_test,verbose=0)
for i in range(10):
    print(pred[i])
plt.plot(history.history['accuracy'], 'r--')
plt.plot(history.history['val_accuracy'], 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.show()
