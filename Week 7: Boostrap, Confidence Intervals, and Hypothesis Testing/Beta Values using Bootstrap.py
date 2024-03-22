# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from randomuniverse import RandomUniverse
%matplotlib inline

# Read the file "Advertising_csv"
df = pd.read_csv('Advertising_adj.csv')

# Take a quick look at the data
df.head()

df.shape

# Define a bootstrap function, which takes as input a dataframe 
# It must output a bootstrapped version of the input dataframe
def bootstrap(df):
    selectionIndex = np.random.randint(0, 200, size = 300)
    new_df = df.iloc[selectionIndex]
    return new_df

# Initialize two empty lists to store the beta values
beta0_list, beta1_list = [],[]

# Choose the number of "parallel" Universes to generate the new dataset
number_of_bootstraps = 500

# Loop through the number of bootstraps
for i in range(number_of_bootstraps):

    # Call the bootstrap function to get a bootstrapped version of the data
    df_new = bootstrap(df)

    # Find the mean of the predictor values i.e. tv
    xmean = np.mean(df_new['tv'])

    # Find the mean of the response values i.e. sales
    ymean = np.mean(df_new['sales'])

    #'X' is the predictor variable given by df_new.tv  
    X = df_new['tv']
    
    #'y' is the reponse variable given by df_new.sales 
    y = df_new['sales']
    
    # Compute the analytical values of beta0 and beta1 using the 
    # equation given in the hints
    beta1 = np.sum((X - xmean) * (y - ymean)) / np.sum((X - xmean)**2)
    beta0 = np.sum(ymean - (beta1 * xmean))

    # Append the calculated values of beta1 and beta0 to the appropriate lists
    beta0_list.append(beta0)
    beta1_list.append(beta1)

# Plot histograms of beta_0 and beta_1 using lists created above 

fig, ax = plt.subplots(1,2, figsize=(15,8))
ax[0].hist(beta0_list, bins=30)
ax[1].hist(beta1_list, bins=30)
ax[0].set_xlabel("beta0 Histogram")
ax[1].set_xlabel("beta1 Histogram")
ax[0].set_ylabel('Frequency')
plt.show();
