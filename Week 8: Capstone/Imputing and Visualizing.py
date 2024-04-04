# Import necessary libraries
# Your code here
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt

# Read the datafile "covid.csv"
df = pd.read_csv('covid.csv')

# Take a quick look at the dataframe
df.head(5)

df.shape

# Check if there are any missing or Null values
df.isnull().sum()

### edTest(test_na) ###

# Find the number of rows with missing values
num_null = df.isnull().any(axis=1).sum()
print("Number of rows with null values:", num_null)
df.head(1)

# kNN impute the missing data
# Use a k value of 5
# Your code here
#create an imputer value that will impute 

#create a list of all the column names
col_name = df.columns.unique().to_list()

#mimic the df with a new variable impute df and make sure to add an index
x = df.drop(columns='Urgency')
y=df[['Urgency']]

x_new = x.copy()
y_new = y.copy()

x_new = pd.DataFrame(KNNImputer(n_neighbors = 5).fit_transform(x_new), columns=x_new.columns)


df_imputed = pd.concat([x_new, y], axis=1)
df_imputed.isnull().sum()

### edTest(test_impute) ###
# Replace the original dataframe with the imputed data, continue to use df for the dataframe
# Your code here
df = df_imputed


# Plot an appropriate graph to answer the following question: Which Age Group has the most urgent need for a hospital bed
# Your code here
urgent_df = df[df['Urgency']==1]
bins = [20,30,40,50,60,70]

plt.hist(urgent_df, bins=bins, edgecolor='black')


# Plot an appropriate graph to answer the following question: Among the following symptoms, which is the most common one for patients with urgent need of hospitalization
# Your code here
symp_df = urgent_df[['cough', 'fever', 'sore_throat', 'fatigue','Urgency']]


symp_counts = symp_df.drop(columns='Urgency').sum()

plt.bar(symp_counts.index, symp_counts.values, color='skyblue')

plt.xlabel('Symptoms')
plt.ylabel('Occurence')
plt.title('Frequency of Symptoms Leading to Urgency')


# Plot an appropriate graph to answer the following question: As compared to patients urgent need of hospitalization patients with no urgency have cough as a more common symptom?
# Your code here
nudf = df[df['Urgency']==0]
nudf = nudf.drop(columns='age')
coughdf = nudf.drop(columns='Urgency').sum()

plt.bar(coughdf.index, coughdf.values)

### edTest(test_split) ###
# Split the data into train and test sets with 70% for training
# Use random state of 60 and set of data as the train split

# Your code here
df_train, df_test = train_test_split(df, test_size=.3, random_state=60)

# Save the train data into a csv called "covid_train.csv"
# Remember to not include the default indices
df_train.to_csv('covid_train.csv', index=False)

# Save the test data into a csv called "covid_test.csv"
# Remember to not include the default indices
df_test.to_csv('covid_test.csv', index=False)
