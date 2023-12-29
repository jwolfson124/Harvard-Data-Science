# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
print("done")

# Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")

# Take a quick look at the dataframe
df.head()

# Define an empty Pandas dataframe to store the R-squared value associated with each 
# predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])

df_results

# For each predictor in the dataframe, call the function "fit_and_plot_linear()"
# from the helper file with the predictor as a parameter to the function

x = df[['TV', 'Radio', 'Newspaper']]
y = df[['Sales']]
df_test = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])

for column in x.columns:
    x = df[[column]]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.8, random_state=42)
    name = x.columns.tolist()
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    train_pred = lr.predict(x_train)
    test_pred = lr.predict(x_test)
    trainr2 = r2_score(y_train, train_pred)  
    testr2 = r2_score(y_test, test_pred) 
        # This function will split the data into train and test split, fit a linear model
        # on the train data and compute the R-squared value on both the train and test data
        # **Your code here**'
    add_row = pd.DataFrame([{'Predictor': name, 'R2 Train': trainr2, 'R2 Test': testr2}])
    df_test = pd.concat([df_test, add_row])
df_test

multireg = pd.DataFrame(fit_and_plot_multi())
multi_train = multireg.iloc[0,0]
multi_test = multireg.iloc[1,0]
print("Train: " , multi_train, "\n Test: ", multi_test)

multi_row = pd.DataFrame([{"Predictor" : "Multi Reg", "R2 Train" : multi_train, "R2 Test": multi_test}])
df_test = pd.concat([df_test, multi_row])
df_results = df_test
# Take a quick look at the dataframe
df_results.head()
