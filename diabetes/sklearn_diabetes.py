from sklearn import datasets, linear_model, model_selection
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.load_diabetes(return_X_y=True)

X_bmi = X[:, np.newaxis, 2] 

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_bmi, y, test_size=0.33, random_state=42)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Real data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='LinearRegression')

plt.xlabel('BMI')
plt.ylabel('Progress ill')
plt.title('linear: BMI dependence')
plt.legend()

plt.savefig('diabetes_plot.png')
print("diabetes_plot.png")
