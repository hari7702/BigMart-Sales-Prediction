#!/usr/bin/env python
# coding: utf-8

# # BigMart Sales Prediction

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


# Step 1: Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


test.head()


# In[6]:


test.info()


# In[7]:


train.isnull()


# In[8]:


test.isnull()


# In[9]:


# Step 2: Data Preprocessing
# Fill Missing Values
train["Item_Weight"].fillna(train["Item_Weight"].median(), inplace=True)
test["Item_Weight"].fillna(test["Item_Weight"].median(), inplace=True)
train["Outlet_Size"].fillna(train["Outlet_Size"].mode()[0], inplace=True)
test["Outlet_Size"].fillna(test["Outlet_Size"].mode()[0], inplace=True)


# In[10]:


# Fix Inconsistent 'Item_Fat_Content'
train["Item_Fat_Content"].replace({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}, inplace=True)
test["Item_Fat_Content"].replace({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}, inplace=True)


# In[11]:


# Create New Features
train["Outlet_Age"] = 2024 - train["Outlet_Establishment_Year"]
test["Outlet_Age"] = 2024 - test["Outlet_Establishment_Year"]
train['Weight_MRP'] = train['Item_Weight'] * train['Item_MRP']
test['Weight_MRP'] = test['Item_Weight'] * test['Item_MRP']


# In[12]:


# Step 3: Encode Categorical Variables
encoder = LabelEncoder()
cat_columns = ["Item_Fat_Content", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Type", "Outlet_Identifier"]
for col in cat_columns:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])


# In[13]:


# Step 4: Prepare Data
X = train.drop(columns=["Item_Outlet_Sales", "Item_Identifier", "Outlet_Establishment_Year"])
y = train["Item_Outlet_Sales"]
test_final = test.drop(columns=["Item_Identifier", "Outlet_Establishment_Year"])


# In[14]:


# Outlier Capping in Target Variable
upper_limit = np.percentile(y, 99)
y = np.clip(y, None, upper_limit)
print("Outliers capped at 99th percentile.")


# In[15]:


# Align train and test data
X, test_final = X.align(test_final, join='inner', axis=1)


# In[16]:


# Step 5: Feature Scaling
scaler = StandardScaler()
numerical_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Weight_MRP']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
test_final[numerical_columns] = scaler.transform(test_final[numerical_columns])


# In[17]:


# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Step 6: Hyperparameter Tuning
param_xgb = {"n_estimators": [300, 400], "learning_rate": [0.01, 0.05], "max_depth": [5, 7]}
xgb = GridSearchCV(XGBRegressor(random_state=42), param_xgb, cv=5, scoring="neg_root_mean_squared_error", verbose=1)
xgb.fit(X_train, y_train)
best_xgb = xgb.best_estimator_


# In[19]:


param_rf = {"n_estimators": [300, 400], "max_depth": [15, 20]}
rf = GridSearchCV(RandomForestRegressor(random_state=42), param_rf, cv=5, scoring="neg_root_mean_squared_error", verbose=1)
rf.fit(X_train, y_train)
best_rf = rf.best_estimator_


# In[20]:


# Feature Importance from Best XGBoost Model

importances = best_xgb.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - XGBoost")
plt.show()


# In[21]:


# Step 7: Stacking Ensemble
# Add Gradient Boosting to Stacking Ensemble
estimators = [
    ('xgb', best_xgb),
    ('rf', best_rf),
    ('gbr', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
]
final_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
final_model.fit(X_train, y_train)


# In[22]:


# Step 8: Validation
y_pred_valid = final_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Validation RMSE: {rmse:.4f}")


# In[23]:


# Step 9: Final Predictions
final_predictions = final_model.predict(test_final)


# In[24]:


# Step 10: Generate Submission File
submission = sample_submission.copy()
submission['Item_Outlet_Sales'] = final_predictions
submission.to_csv('final_submission_optimized.csv', index=False)
print("Final submission file saved as 'final_submission_optimized.csv'")


# In[ ]:




