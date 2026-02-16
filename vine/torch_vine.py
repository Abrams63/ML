import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets, model_selection, preprocessing, metrics
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

X, y = datasets.load_wine(return_X_y=True)
target_names = datasets.load_wine().target_names

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

class WineClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WineClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

model = WineClassifier(input_dim=13, output_dim=3)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    _, y_pred_t = torch.max(logits, 1)
    y_pred = y_pred_t.numpy()

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
ax.set_title("Confusion Matrix (PyTorch Model)")
plt.savefig('wine_torch_confusion_matrix.png')
print("Save 'wine_torch_confusion_matrix.png'")