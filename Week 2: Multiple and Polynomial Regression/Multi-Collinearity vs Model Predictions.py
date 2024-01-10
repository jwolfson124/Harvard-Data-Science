# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns 
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
%matplotlib inline
print("Done")

# Read the file named "colinearity.csv" into a Pandas dataframe
df = pd.read_csv("colinearity.csv")

# Take a quick look at the dataset
df.head()

# Choose all the predictors as the variable 'X' (note capitalization of X for multiple features)
X = df.drop('y', axis=1)

# Choose the response variable 'y' 
y = df['y'].values

### edTest(test_coeff) ###

# Initialize a list to store the beta values for each linear regression model
linear_coef = []

# Loop over all the predictors
# In each loop "i" holds the name of the predictor 
for var in X:
    
    # Set the current predictor as the variable x
    x = df[[var]]

    # Create a linear regression object
    linreg = LinearRegression()

    # Fit the model with training data 
    # Remember to choose only one column at a time i.e. given by x (not X)
    linreg.fit(x,y)
    
    # Add the coefficient value of the model to the list
    linear_coef.append(linreg.coef_)
linear_coef

### edTest(test_multi_coeff) ###

# Perform multi-linear regression with all predictors
multi_linear = LinearRegression()

# Fit the multi-linear regression on all features of the entire data
multi_linear.fit(X,y)

# Get the coefficients (plural) of the model
multi_coef = multi_linear.coef_

# Helper code to see the beta values of the linear regression models

print('By simple(one variable) linear regression for each variable:', sep = '\n')

for i in range(4):
    print(f'Value of beta{i+1} = {linear_coef[i][0]:.2f}')

# Helper code to compare with the values from the multi-linear regression
print('By multi-Linear regression on all variables')
b=0
while b < len(multi_coef):
    print("Value of Beta",b,(multi_coef[b]))
    b += 1

# Helper code to visualize the heatmap of the covariance matrix
corrMatrix = df[['x1','x2','x3','x4']].corr() 
sns.heatmap(corrMatrix, annot=True) 
plt.show()
