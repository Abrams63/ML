import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=3)

model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(3, 
                 activation=None, 
                 kernel_regularizer=regularizers.l2(0.01)) 
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='categorical_hinge', metrics=['accuracy'])


model.fit(X_train, y_train_oh, epochs=100, batch_size=16, verbose=0)

y_pred_logits = model.predict(X_test)
y_pred = np.argmax(y_pred_logits, axis=1)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8, 6))
metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    display_labels=target_names, 
    cmap=plt.cm.Blues,
    ax=ax
)
ax.set_title("Confusion Matrix (TensorFlow Model)")
plt.savefig('iris_tf_confusion_matrix.png')
print("Save 'iris_tf_confusion_matrix.png'")