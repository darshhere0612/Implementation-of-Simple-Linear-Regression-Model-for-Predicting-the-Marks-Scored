## Implementation of Simple Linear Regression Model for Predicting the Marks Scored

### AIM

To write and verify a Python program to implement a Simple Linear Regression model for predicting the marks scored by a student based on study hours.

### Equipments Required

* **Hardware:** PC / Laptop
* **Software:** Anaconda / Python 3.7+ Installation, Jupyter Notebook (with `numpy`, `matplotlib`, and `scikit-learn`)

---

### Algorithm

1. Collect or define the dataset containing the independent variable $X$ (e.g., Hours Studied) and the dependent variable $Y$ (e.g., Marks Scored).
2. Split the dataset into training and testing sets (or use all data for fitting a simple model).
3. Compute the mean of $X$ ($\bar{X}$) and the mean of $Y$ ($\bar{Y}$):

$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad \bar{Y} = \frac{1}{n}\sum_{i=1}^{n} y_i$$


4. Calculate the slope $m$ and intercept $b$ using ordinary least squares:

$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})}{\sum_{i=1}^{n} (x_i - \bar{X})^2}, \qquad b = \bar{Y} - m\bar{X}$$


5. Form the linear regression equation $\hat{Y} = mX + b$ to predict marks for new study hour values.
6. Evaluate the model and visualize the actual data points along with the predicted regression line.

---

### Program

```python
"""
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Darshan J
RegisterNumber: 212224020008
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset: X = Hours Studied, Y = Marks Scored
hours = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]).reshape(-1, 1)
marks = np.array([35, 40, 45, 50, 54, 60, 62, 68, 72, 75, 80, 84, 88, 92, 95])

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(hours, marks, test_size=0.2, random_state=42)

# 3. Train the Simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
b = model.intercept_

# 4. Predict marks on test set and for a new student
y_pred = model.predict(X_test)
new_hours = np.array([[5.8]])
predicted_mark = model.predict(new_hours)[0]

# 5. Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Regression Equation: Marks = {m:.4f} * Hours + {b:.4f}")
print(f"Predicted Marks for {new_hours[0][0]} hours of study: {predicted_mark:.2f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. Visualize actual data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(hours, marks, color="blue", label="Actual Marks")
plt.plot(hours, model.predict(hours), color="red", linewidth=2, label="Regression Line")
plt.scatter(new_hours, [predicted_mark], color="green", s=80, zorder=5, label=f"Prediction ({new_hours[0][0]}h -> {predicted_mark:.1f})")
plt.title("Simple Linear Regression: Study Hours vs. Marks Scored")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.legend()
plt.grid(True)
plt.show()

```

---

### Output

```text
Slope (m): 8.5455
Intercept (b): 28.4545
Regression Equation: Marks = 8.5455 * Hours + 28.4545
Predicted Marks for 5.8 hours of study: 78.02
Mean Squared Error (MSE): 2.1488
R² Score: 0.9922

```

---

### Result

Thus, the program to implement the simple linear regression model for predicting the marks scored was written and verified using Python programming.
