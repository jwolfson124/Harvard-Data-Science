import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
print("done")

df = pd.read_csv()
# The response variable will be 'Balance.'
x = df.drop('Balance', axis=1)
y = df['Balance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Trying to fit on all features in their current representation throws an error.
try:
    test_model = LinearRegression().fit(x_train, y_train)
except Exception as e:
    print('Error!:', e)

# Inspect the data types of the DataFrame's columns.
df.dtypes

# Fit a linear model using only the numeric features in the dataframe.
df_nobal = df.drop('Balance', axis=1)
types = pd.DataFrame(df_nobal.dtypes)
numeric_features = []
for column, dtype in df_nobal.dtypes.items():
    if dtype in ("int64", "float64"):
        numeric_features.append(column)
    else:
        pass

numeric_features
model1 = LinearRegression().fit(x_train[numeric_features], y_train)

# Report train and test R2 scores.
train_score = model1.score(x_train[numeric_features], y_train)
test_score = model1.score(x_test[numeric_features], y_test)
print('Train R2:', train_score)
print('Test R2:', test_score)

print('In the train data, Ethnicity takes on the values:', list(x_train['Ethnicity'].unique()))

dummy_var = pd.get_dummies(df["Ethnicity"], drop_first=True)

df_dummies = df.drop('Balance', axis=1)
types = pd.DataFrame(df_dummies.dtypes)
features = []
hot = []
for column, dtype in df_dummies.dtypes.items():
    if dtype in ("int64", "float64"):
        features.append(df_dummies[column])
    else:
        item = pd.get_dummies(df_dummies[column], drop_first=True, prefix=column)
        hot.append(item)
        hot_df = pd.concat(hot)
        
        

        
features = pd.DataFrame(features).T
hot_df = pd.concat(hot, axis=1)
features_df = pd.concat([features, hot_df], axis=1)
features_df

x_train, x_test = train_test_split(features_df, test_size =.2, random_state = 42)
x_train_design = x_train
x_test_design = x_test
x_train_design.head()

# Fit model2 on design matrix
model2 = LinearRegression().fit(x_train_design, y_train)

# Report train and test R2 scores
train_score = model2.score(x_train_design, y_train)
test_score = model2.score(x_test_design, y_test)
print('Train R2:', train_score)
print('Test R2:', test_score)

# Note that the intercept is not a part of .coef_ but is instead stored in .intercept_.
coefs = pd.DataFrame(model2.coef_, index=x_train_design.columns, columns=['beta_value'])
coefs

# Visualize crude measure of feature importance.
sns.barplot(data=coefs.T, orient='h').set(title='Model Coefficients');

# Specify best categorical feature
best_cat_feature = 'Student_Yes'

# Define the model.
featurex = ['Income', best_cat_feature]
model3 = LinearRegression()
model3.fit(x_train_design[featurex], y_train)

# Collect betas from fitted model.
beta0 = model3.intercept_
beta1 = model3.coef_[featurex.index('Income')]
beta2 = model3.coef_[featurex.index(best_cat_feature)]

# Display betas in a DataFrame.
coefs = pd.DataFrame([beta0, beta1, beta2], index=['Intercept']+featurex, columns=['beta_value'])
print(coefs)
print("0", beta0)
print("1", beta1)
print("2", beta2)

# Create space of x values to predict on.
x_space = np.linspace(x['Income'].min(), x['Income'].max(), 1000)

# Generate 2 sets of predictions based on best categorical feature value.
# When categorical feature is true/present (1)
y_hat_yes = beta0 + beta1 * x_space + beta2 * 1
# When categorical feature is false/absent (0)
y_hat_no = beta0 + beta1 * x_space + beta2 * 0

# Plot the 2 prediction lines for students and non-students.
ax = sns.scatterplot(data=pd.concat([x_train_design, y_train], axis=1), x='Income', y='Balance', hue=best_cat_feature, alpha=0.8)
ax.plot(x_space, y_hat_no)
ax.plot(x_space, y_hat_yes);
