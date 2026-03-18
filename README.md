# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries (numpy, matplotlib).

2.Define dataset values x and y.

3.Initialize parameters m, b, learning rate, and epochs.

4.Apply gradient descent to update m and b.

5.Print results and plot the regression line.

## Program:
```
 import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
m = 0  # slope
b = 0  # intercept

learning_rate = 0.01
epochs = 1000
n = len(x)

# Gradient Descent
for i in range(epochs):
    
    # Predicted values
    y_pred = m * x + b
    
    # Calculate gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Plot results
y_pred = m * x + b

plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, y_pred, color='red', label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
/*
Program to implement the linear regression using gradient descent.
Developed by: Siva Sundar P
RegisterNumber:  25011320
*/
```

## Output:
<img width="461" height="302" alt="image" src="https://github.com/user-attachments/assets/ff8267b4-a65b-48b6-8051-110afb38ffff" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
