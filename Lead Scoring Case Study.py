#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Define file paths
data_path = r'C:\Users\KOMM\OneDrive - COWI\Desktop\Learning\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv'
data_dict_path = r'C:\Users\KOMM\OneDrive - COWI\Desktop\Learning\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads Data Dictionary.xlsx'
lead_scores_path = r'C:\Users\KOMM\OneDrive - COWI\Desktop\Learning\Lead+Scoring+Case+Study\Lead Scoring Assignment\lead_scores.csv'

# Load the dataset
data = pd.read_csv(data_path)

# Load and review the data dictionary
data_dict = pd.read_excel(data_dict_path)

# Print the data dictionary to understand the columns
print("Data Dictionary:")
print(data_dict)

# Print initial dataset information
print("\nInitial Data Overview:")
print(data.head())
print(data.info())
print(data.describe())

# Data Cleaning
# Replace 'Select' with NaN
data.replace('Select', pd.NA, inplace=True)

# Fill missing values for numeric columns with median
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].median(), inplace=True)

# Fill missing values for categorical columns with 'Unknown'
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna('Unknown', inplace=True)

# Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data_encoded.drop('Converted', axis=1)
y = data_encoded['Converted']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_prob)}')

# Assign Lead Scores (between 0 and 100)
lead_scores = (y_prob * 100).astype(int)

# Create a DataFrame to save lead scores
results = pd.DataFrame({'LeadID': X_test.index, 'LeadScore': lead_scores})

# Save the lead scores to a CSV file
results.to_csv(lead_scores_path, index=False)

# Load and print the contents of the lead_scores.csv file
saved_results = pd.read_csv(lead_scores_path)
print("\nSaved Lead Scores:")
print(saved_results.head())


# In[ ]:




