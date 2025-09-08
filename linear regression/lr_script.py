# import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppl

# import data set

dataset = pd.read_csv("Salary_Data.csv")
experience = dataset.iloc[:, :-1].values
salary = dataset.iloc[:, -1].values

#split dataset into training set and test set

from sklearn.model_selection import train_test_split
experience_train, experience_test, salary_train, salary_test = train_test_split(experience, salary, test_size=0.2, random_state=0)

# Trainning simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(experience_train, salary_train)

# Predicting the test set results
salary_pred = regressor.predict(experience_test)

# Visualising the training set results
ppl.scatter(experience_train, salary_train, color='red')
ppl.plot(experience_train, regressor.predict(experience_train), color='blue')
ppl.title("Salary vs Experience (Training set)")
ppl.xlabel("Years of Experience")
ppl.ylabel("Salary")
ppl.show()

# Visualising the test set results
ppl.scatter(experience_test, salary_test, color='red')
ppl.plot(experience_train, regressor.predict(experience_train), color='blue')
ppl.title("Salary vs Experience (Test set)")
ppl.xlabel("Years of Experience")
ppl.ylabel("Salary")
ppl.show()

print(regressor.predict([[12]]))
