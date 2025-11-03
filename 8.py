from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
max_features = 10000
maxlen = 500          
batch_size = 128
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), "train sequences")
print(len(x_test), "test sequences")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
model = Sequential([Embedding(max_features, 32),SimpleRNN(32),    Dense(1, activation='sigmoid')])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
plt.plot(history.history['accuracy'], 'bo', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'b-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 'ro', label='Training Loss')
plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
