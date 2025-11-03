import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
(train_x, train_y), (test_x, test_y) = mnist.load_data()
plt.imshow(train_x[0], cmap='gray')
train_x = train_x.reshape(-1, 28, 28, 1) / 255.0
test_x = test_x.reshape(-1, 28, 28, 1) / 255.0
train_y, test_y = to_categorical(train_y), to_categorical(test_y)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=13)
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)),
    LeakyReLU(0.1),
    MaxPooling2D((2,2), padding='same'),
    Flatten(),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=3, batch_size=64, verbose=1)
loss, acc = model.evaluate(test_x, test_y, verbose=0)
print(f"\nTest Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")
plt.plot(history.history['accuracy'], '--', label='Train')
plt.plot(history.history['val_accuracy'], 'b', label='Val')
plt.title('Accuracy'); plt.legend()
plt.show()
plt.plot(history.history['loss'], '--', label='Train')
plt.plot(history.history['val_loss'], 'b', label='Val')
plt.title('Loss'); plt.legend()
plt.show()
plt.show()
