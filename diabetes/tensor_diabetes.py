import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

data = load_diabetes()
X, y = data.data, data.target

scaler = StandardScaler()

X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(32, activation='relu'),
    
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mse',
              metrics=['mae'])

history = model.fit(X_train, y_train,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    verbose=1)


def plot_history(history):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], 'b-', label='Training loss')
    plt.plot(epochs, hist['val_loss'], 'r--', label='Validation loss')
    plt.title('Loss (MSE)')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['mae'], 'b-', label='Training MAE')
    plt.plot(epochs, hist['val_mae'], 'r--', label='Validation MAE')
    plt.title('MAE')
    plt.xlabel('Эпохи')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('diabetes_plot.png')
    plt.show() # Добавил show(), чтобы график отобразился сразу

# Вызов функции
plot_history(history)