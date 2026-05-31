# Use this in Google Colab as .ipynb file
# Part 1: Data Understanding and Preparation
## 1. First, I loaded the dataset. Then, in order for the file to read the dataset, I moved the file into /content working directory.
## 2. Each row represents an individual insurance customer. The row includes the customer's age, sex, BMI, number of children, whether the customer is a smoker or not, the region they're located in, and the cost of charges.
## 3. The target variable is "charges." This dependent variable represents the medical charges for the customer, which is an outcome of the other input values in the dataset.
## 4. After inspecting the dataset's data types, missing values, and anomalies, there are no issues with any of the aforementioned.

import pandas as pd # importing pandas library and creating an alias of pandas called pd

df = pd.read_csv("insurance.csv") # reading "insurance.csv" using pandas library

y = df["charges"]          # extracts the target variable, "charges"
X = df.drop("charges", axis=1)   # removes "charges" from the dataset for the WHOLE COLUMN (axis=1)

# One-hot encoding
# Converts variable into binary indicator variables
# get_dummies is used for "converting categorical variables into dummy/indicator variables"
# if value is represented then true, else false
# Using drop_first=True to drop 1 table and use as a reference baseline. Prevents perfect multicollinearity
X_encoded = pd.get_dummies(X, columns=["sex", "smoker", "region"], drop_first=True)

X_encoded = X_encoded.astype(int) # casts the data as an int

print(X_encoded.shape) # .shape for row and column count
print(y.shape)

# Output
# (1338, 8) --> 1338 samples, 8 predictor variables
# (1338,) --> 1 target

# Part 2: Supervised Model (Regression)
## Fit an Ordinary Least Squares (OLS) regression model using the prepared data. Purpose: to predict the target variable, "charges."

import statsmodels.api as sm # For statistical modeling

X_ols = sm.add_constant(X_encoded)
model = sm.OLS(y, X_ols).fit()

# Sanity Check
# Ensures that any changes haven't broken the code
X_ols.dtypes #dtypes returns the data types

# Overall summary of the OLS Regression Results
model.summary()

# Report estimated numeric coefficients for each variable
model.params

# Report r-squared
# Purpose of r-squared: percentage of variance of charges explained by the model
print("r-squared: ", model.rsquared)

# Interpret at least two coefficients in plain language
## Smoker coefficient: The smoker coefficient (smoker_yes) is approximately 23859.78. This means that as long as the other variables remain the same, being a smoker raises the expected medical charges by approximately $23,859.79.
## BMI coefficient: The BMI coefficient is approximately 337.99. This means that if the BMI increases by one point, the expected medical charges will increase by approximately $337.99.

# Explain what the model claims about the relationship between inputs and the target
## Based off the OLS Regression Results, the model is claiming that smoking contributes the most towards a higher medical cost. The model also claims that variables like age and BMI have positive effects towards the overall medical cost. So, being older or having a higher BMI will increase a customer's medical cost. Variables like the number of children and region have little to no influence on the overall medical cost.

# Sources/Citations
##Importing pandas library: https://www.w3schools.com/python/pandas/pandas_getting_started.asp
## Reading files using pandas: https://www.w3schools.com/python/pandas/pandas_csv.asp
## One-hot encoding: https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
## get_dummies: https://www.geeksforgeeks.org/pandas/python-pandas-get_dummies-method/
## Dropping columns: https://stackoverflow.com/questions/63661560/drop-first-true-during-dummy-variable-creation-in-pandas
## Data types: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html
## Sanity check: https://www.reddit.com/r/QualityAssurance/comments/1dqpx39/can_someone_please_explain_the_difference_between/
## .shape (Sanity check): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
## Casting (.astype): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
## OLS Model in Python: https://www.youtube.com/watch?v=bw_wBbUKJW8
## Importing statsmodels.api module: https://www.statsmodels.org/stable/index.html
## Reporting the estimated coefficients: https://www.geeksforgeeks.org/data-analysis/extracting-regression-coefficients-from-statsmodelsapi/
## R-squared: https://www.reddit.com/r/statistics/comments/1cwvp5w/q_how_to_interpret_an_r2_value/#:~:text=R2%20is%20the%20percentage,predicted%20values%20were%20perfectly%20accurate.
