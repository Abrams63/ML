import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

data = load_diabetes()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

print("\nЗапуск обучения...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("\nОценка на тестовом наборе:")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

predictions = model.predict(X_test[:5])
print("\nПримеры предсказаний vs Реальные значения:")
for pred, actual in zip(predictions, y_test[:5]):
    print(f"Предсказано: {pred[0]:.2f}, Реально: {actual[0]:.2f}")

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

    # График MAE
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

plot_history(history)