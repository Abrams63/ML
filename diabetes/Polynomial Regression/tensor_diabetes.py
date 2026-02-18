import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1) 
y = data.target.reshape(-1, 1)

degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(1, input_shape=(X_train.shape[1],)) 
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

print(f"\nTest MSE: {model.evaluate(X_test, y_test, verbose=0)[0]:.4f}")

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
x_range_scaled = scaler.transform(x_range_poly)
y_range_pred = model.predict(x_range_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Данные')
plt.plot(x_range, y_range_pred, color='red', linewidth=3, label=f'Polynomial Regression (deg={degree})')
plt.title('TensorFlow: Polynomial Regression on Diabetes (BMI)')
plt.xlabel('BMI (normalized)')
plt.ylabel('Disease Progression')
plt.legend()
plt.grid(True)
plt.savefig('tf_polynomial_plot.png')
print("Save tf_polynomial_plot.png")