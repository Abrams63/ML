from sklearn import datasets, linear_model, model_selection
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.load_diabetes(return_X_y=True)
X_bmi = X[:, np.newaxis, 2] 

degree = 2 
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X_bmi)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_poly, y, test_size=0.33, random_state=42
)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

x_range = np.linspace(X_bmi.min(), X_bmi.max(), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X_bmi, y, color='black', alpha=0.3, label='Real data')
plt.plot(x_range, y_range_pred, color='red', linewidth=3, label=f'Polynomial (degree={degree})')

plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title(f'Polynomial Regression: Degree {degree}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('diabetes_poly_plot.png')
print("diabetes_poly_plot.png saved")