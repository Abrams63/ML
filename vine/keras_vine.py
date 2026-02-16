import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model # type: ignore
from sklearn import datasets, model_selection, preprocessing, metrics
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.load_wine(return_X_y=True)
target_names = datasets.load_wine().target_names

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


inputs = Input(shape=(13,), name="wine_features")  
outputs = layers.Dense(3, name="predictions")(inputs) 

model = Model(inputs=inputs, outputs=outputs, name="wine_functional_model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

model.fit(X_train, y_train, epochs=200, verbose=1)


logits = model.predict(X_test)
y_pred = np.argmax(logits, axis=1)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8, 6))
metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    display_labels=target_names, 
    cmap=plt.cm.Oranges,
    ax=ax
)
ax.set_title("Confusion Matrix (Keras Functional API)")
plt.savefig('wine_keras_functional_matrix.png')
print("Save 'wine_keras_functional_matrix.png'")