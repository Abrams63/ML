import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets, model_selection

X, y = datasets.load_diabetes(return_X_y=True)
X_bmi = X[:, np.newaxis, 2]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_bmi, y, test_size=0.33, random_state=42
)


X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1) 

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

model = nn.Linear(1, 1)


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

num_epochs = 5000 
loss_history = []

for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():
    predicted = model(X_test_tensor).detach().numpy()

plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='black', label='Real data')

plt.plot(X_test, predicted, color='purple', linewidth=3, label='PyTorch Linear Regression')

plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('PyTorch: BMI dependence')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('diabetes_plot_pytorch.png')
print("diabetes_plot_pytorch.png saved")