import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC 

import matplotlib
matplotlib.use('Agg')

iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearSVC(C=1.0, penalty='l2', loss='squared_hinge', max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# 5. Предсказание
y_pred = model.predict(X_test)

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
ax.set_title("Confusion Matrix (scikit-learn LinearSVC)")
plt.savefig('iris_sklearn_confusion_matrix.png')
print("Saved 'iris_sklearn_confusion_matrix.png'")