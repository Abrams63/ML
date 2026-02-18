import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets, model_selection
from sklearn.preprocessing import PolynomialFeatures

X, y = datasets.load_diabetes(return_X_y=True)
X_bmi = X[:, np.newaxis, 2]

degree = 2 
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X_bmi) 

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_poly, y, test_size=0.33, random_state=42
)

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1) 

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)


model = nn.Linear(degree, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


num_epochs = 10000
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    x_range = np.linspace(X_bmi.min(), X_bmi.max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    x_range_tensor = torch.from_numpy(x_range_poly).float()
    y_range_pred = model(x_range_tensor).numpy()

plt.figure(figsize=(10, 6))

indices = model_selection.train_test_split(np.arange(len(y)), test_size=0.33, random_state=42)[1]
plt.scatter(X_bmi[indices], y_test, color='black', alpha=0.5, label='Real data (Test)')

plt.plot(x_range, y_range_pred, color='red', linewidth=3, label=f'Polynomial Regression (deg={degree})')

plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title(f'PyTorch: Polynomial Regression with Adam (Degree {degree})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('diabetes_polynomial.png')
print("diabetes_polynomial.png saved")