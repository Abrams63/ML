import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

class LinearSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

model = LinearSVM(4, 3)
criterion = nn.MultiMarginLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.numpy()
    y_true = y_test.numpy()

accuracy = metrics.accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(metrics.classification_report(y_true, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8, 6))
metrics.ConfusionMatrixDisplay.from_predictions(
    y_true, 
    y_pred, 
    display_labels=target_names, 
    cmap=plt.cm.Blues,
    ax=ax
)
ax.set_title("Confusion Matrix (PyTorch Model)")
plt.savefig('iris_torch_confusion_matrix.png')
print("Save 'iris_torch_confusion_matrix.png'")