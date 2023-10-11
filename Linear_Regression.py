# -*- coding: utf-8 -*-
"""
Spyder Editor

Machine Learning
Owned by: Erdem Taha Sokullu

This is a temporary script file.
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
df = pd.read_csv("linear_regression_dataset.csv", sep=";")

# Plot Data
plt.scatter(df.experience, df.salary)
plt.xlabel("experience")
plt.ylabel("salary")

# Using the Sklearn Library
from sklearn.linear_model import LinearRegression

# Linear Regression Model
linear_reg = LinearRegression()
# Note 1: These structures are in Pandas, but in sklearn, numpy is more functional, so we need to convert them to numpy.
# To do this, we add .values to the data.
# Note 2: If Pandas shape is (14,), sklearn does not accept this; instead, we need to reshape it to (14,1).
x = df.experience.values.reshape(-1, 1)
y = df.salary.values.reshape(-1, 1)

# Linear Regression Fit
linear_reg.fit(x, y)

import numpy as np

# Linear Regression Predict
b0 = linear_reg.predict([[0]])  # This gives us the intercept point on the y-axis (intercept) # 1163

b0_ = linear_reg.intercept_  # This command gives the intersection point # 1163

b1 = linear_reg.coef_  # b1 gives the slope # 1138

# salary = 1663 + 1138 * experience (y = b0 + b1*x)
trial = 1663.89519747 + 1138.34819698 * 10
print(trial)

print(linear_reg.predict([[11]]))
# The predict command gives the corresponding answer for the desired x value using the equation above

array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1) # Desired experience list

plt.scatter(x, y, color="green")  # Command to create a scatter plot
plt.show()

y_head = linear_reg.predict(array)  # Determines salary values corresponding to the desired experience list
plt.plot(array, y_head, color="red")  # This creates a line according to our salary and experience matches