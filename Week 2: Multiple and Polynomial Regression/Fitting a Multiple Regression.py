# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
import itertools
print("Done")

# Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")
# Take a quick look at the data to list all the predictors
df.head()

# Initialize a list to store the MSE values
#mse_df = pd.DataFrame(columns = ['Predictor',  'MSE'])


# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would 
# end up with [['A'],['B'],['A','B']]
#cols = [['TV'], ['Radio'], ['Newspaper'],['TV', 'Radio'], ['TV', 'Newspaper'],  ['Radio', 'Newspaper'],  ['TV', 'Radio', 'Newspaper']]
# Loop over all the predictor combinations 
#for i in cols:

    # Set each of the predictors from the previous list as x
#    x = df[i]
    
    # Set the "Sales" column as the reponse variable
#    y = df[['Sales']]
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
 #   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .8, random_state = 0)

    # Initialize a Linear Regression model
#    lreg = LinearRegression()

    # Fit the linear model on the train data
#    lreg.fit(x_train, y_train)

    # Predict the response variable for the test set using the trained model
#    y_pred_test = lreg.predict(x_test)

    # Compute the MSE for the test data
#    MSE_test = mean_squared_error(y_test, y_pred_test).tolist()

#    predictor = x.columns.tolist()
    # Append the computed MSE to the initialized list
 #   new_row = pd.DataFrame([{'Predictor' : predictor, 'MSE': MSE_test}])
  #  mse_df = pd.concat([mse_df, new_row])
#mse_df

### edTest(test_mse) ###

# Initialize a list to store the MSE values
mse_list = []


# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would 
# end up with [['A'],['B'],['A','B']]
cols = [['TV'], ['Radio'], ['Newspaper'],['TV', 'Radio'], ['TV', 'Newspaper'],  ['Radio', 'Newspaper'],  ['TV', 'Radio', 'Newspaper']]
# Loop over all the predictor combinations 
for i in cols:

    # Set each of the predictors from the previous list as x
    x = df[i]
    
    # Set the "Sales" column as the reponse variable
    y = df[['Sales']]
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train, y_train)

    # Predict the response variable for the test set using the trained model
    y_pred_test = lreg.predict(x_test)

    # Compute the MSE for the test data
    MSE_test = mean_squared_error(y_test, y_pred_test).tolist()

    predictor = x.columns.tolist()
    # Append the computed MSE to the initialized list
    mse_list.append(MSE_test)
mse_list

# Helper code to display the MSE for each predictor combination
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],round(mse_list[i],3)])

print(t)

