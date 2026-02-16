from sklearn import datasets, linear_model, model_selection, preprocessing, metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X, y = datasets.load_wine(return_X_y=True)
feature_names = datasets.load_wine().feature_names
target_names = datasets.load_wine().target_names

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = linear_model.LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"(Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8, 6))
cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    display_labels=target_names, 
    cmap=plt.cm.Blues,
    ax=ax
)
ax.set_title("Confusion Matrix")

plt.savefig('wine_confusion_matrix.png')
print("Save 'wine_confusion_matrix.png'")