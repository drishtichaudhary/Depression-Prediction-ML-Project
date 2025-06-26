#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd

# Define the file path
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Projects\final_depression_dataset_1.csv"


# In[20]:


# Load the dataset
df = pd.read_csv(file_path)


# In[21]:


# Display basic info
print(df.info())
print(df.head())  # Show first 5 rows


# In[22]:


# Check missing values
print(df.isnull().sum())


# In[23]:


# Drop columns with too many missing values
df.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction'], inplace=True)


# In[24]:


# Fill numerical missing values with median (without inplace=True)
df['Work Pressure'] = df['Work Pressure'].fillna(df['Work Pressure'].median())
df['Job Satisfaction'] = df['Job Satisfaction'].fillna(df['Job Satisfaction'].median())


# In[25]:


# Fill categorical missing values with mode (without inplace=True)
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])


# In[26]:


# Verify if all missing values are handled
print(df.isnull().sum())  


# In[27]:


from sklearn.preprocessing import LabelEncoder

# Binary Encoding (Yes/No → 0/1)
binary_cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Depression']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})


# In[28]:


# Label Encoding for smaller categorical features
label_cols = ['Gender', 'Working Professional or Student', 'Sleep Duration', 'Dietary Habits']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])


# In[29]:


# One-Hot Encoding for larger categorical features
df = pd.get_dummies(df, columns=['City', 'Profession', 'Degree'], drop_first=True)  # Avoids multicollinearity

# Label Encoding	When the categorical variable has an order/rank (e.g., Small < Medium < Large).
#One-Hot Encoding	When the categorical variable has no inherent order (e.g., City, Country, Profession).


# In[30]:


# Check dataset after encoding
print(df.head())  


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

# Countplot to visualize class imbalance
sns.countplot(x=df['Depression'])
plt.title("Class Distribution in 'Depression' Column")
plt.show()


# In[32]:


# Print value counts
print(df['Depression'].value_counts())


# In[33]:


#SMOTE (Synthetic Minority Oversampling Technique) creates synthetic examples for the minority class instead of just duplicating them.
from imblearn.over_sampling import SMOTE

# Define SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Adjust ratio as needed


# In[35]:


X_final = df.drop(columns=['Depression', 'Name'], errors='ignore')
y_final = df['Depression']


# In[36]:


# Apply SMOTE
from imblearn.over_sampling import SMOTE


# In[37]:


smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_final, y_final)


# In[39]:


# Convert back to DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_final.columns), pd.DataFrame(y_resampled, columns=['Depression'])], axis=1)


# In[40]:


# Check new class distribution
print(df_resampled['Depression'].value_counts())


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot class distribution
sns.countplot(x=y_resampled, palette='coolwarm')
plt.title("Class Distribution of Depression (After Balancing)")
plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


# In[42]:


# Plot histograms for all numerical features
X_resampled.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions (Histograms)", fontsize=16)
plt.show()


# In[43]:


# Select numerical columns only
num_cols = X_resampled.select_dtypes(include=['int64', 'float64']).columns

# Plot boxplots in chunks
chunk_size = 5  # Number of features per plot
for i in range(0, len(num_cols), chunk_size):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=X_resampled[num_cols[i:i + chunk_size]])
    plt.xticks(rotation=90)
    plt.title(f"Boxplot for Features {i+1} to {i+chunk_size}")
    plt.show()


# In[44]:


# Compute the correlation matrix
corr_matrix = X_resampled.corr()


# In[45]:


# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[46]:


import numpy as np

# Function to detect outliers using IQR
def detect_outliers_iqr(df):
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]):  # Select numerical features only
        Q1 = df[col].quantile(0.25)  # First quartile
        Q3 = df[col].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    
    return outlier_counts


# In[47]:


# Detect outliers
outliers_iqr = detect_outliers_iqr(X_resampled)
print("Outlier counts per feature (IQR method):")
print(outliers_iqr)


# In[48]:


# Drop 'Name' and 'City' (if still present)
X_resampled = X_resampled.drop(columns=['Name'], errors='ignore')


# In[49]:


# Check the remaining features
print("Remaining Features:", X_resampled.columns)


# In[50]:


from sklearn.preprocessing import StandardScaler

# Selecting numerical features
num_features = ['Age', 'Work/Study Hours', 'Financial Stress',
                'Sleep Duration', 'Job Satisfaction', 'Work Pressure']


# In[51]:


scaler = StandardScaler()
X_final[num_features] = scaler.fit_transform(X_final[num_features])


# In[52]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Initialize Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_final, y_final)


# In[54]:


# Get feature importance
feature_importances = pd.Series(rf.feature_importances_, index=X_final.columns)
feature_importances.sort_values(ascending=False, inplace=True)


# In[55]:


# Display top features
print(feature_importances.head(10))


# In[58]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)


# In[59]:


# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[60]:


# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)


# In[61]:


# Predict on test data
y_pred = log_reg.predict(X_test_scaled)


# In[62]:


# Evaluate model performance
log_reg_results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred)
}


# In[63]:


log_reg_results


# In[64]:


# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[65]:


# Train the model
rf_model.fit(X_train, y_train)


# In[66]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)


# In[67]:


# Evaluate the model
rf_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1 Score': f1_score(y_test, y_pred_rf)
}


# In[68]:


rf_metrics


# In[69]:


pip install xgboost


# In[70]:


from xgboost import XGBClassifier
# Initialize and train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


# In[71]:


# Make predictions
y_pred_xgb = xgb_model.predict(X_test)


# In[72]:


# Evaluate performance
xgb_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1 Score': f1_score(y_test, y_pred_xgb)
}


# In[73]:


xgb_metrics


# In[74]:


import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Logistic Regression", "Random Forest", "XGBoost"]

# Performance metrics
accuracy = [0.9219, 0.9082, 0.9199]
precision = [0.7742, 0.8235, 0.7907]
recall = [0.7912, 0.6154, 0.7473]
f1_score = [0.7826, 0.7044, 0.7684]

# Bar width and positions
x = np.arange(len(models))
width = 0.2

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width*1.5, accuracy, width, label="Accuracy", color='royalblue')
ax.bar(x - width/2, precision, width, label="Precision", color='seagreen')
ax.bar(x + width/2, recall, width, label="Recall", color='orange')
ax.bar(x + width*1.5, f1_score, width, label="F1 Score", color='red')

# Labels and formatting
ax.set_xlabel("Models")
ax.set_ylabel("Scores")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()


# In[75]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # L1 (Lasso), L2 (Ridge)
    'solver': ['liblinear']  # 'liblinear' supports L1 & L2 penalties
}


# In[76]:


# Initialize Logistic Regression
log_reg = LogisticRegression(random_state=42)


# In[77]:


# Perform Grid Search
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[78]:


# Best parameters
print("Best Parameters:", grid_search.best_params_)


# In[79]:


# Train the best model
best_log_reg = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42)
best_log_reg.fit(X_train, y_train)


# In[80]:


# Predictions
y_pred = best_log_reg.predict(X_test)


# In[81]:


from sklearn import metrics

optimized_metrics = {
    "Accuracy": metrics.accuracy_score(y_test, y_pred),
    "Precision": metrics.precision_score(y_test, y_pred),
    "Recall": metrics.recall_score(y_test, y_pred),
    "F1 Score": metrics.f1_score(y_test, y_pred)  # Avoids conflict
}


# In[82]:


optimized_metrics


# In[83]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get predictions for both train & test sets
y_train_pred = best_log_reg.predict(X_train)  # Predictions on train data
y_test_pred = best_log_reg.predict(X_test)    # Predictions on test data


# In[84]:


# Calculate metrics for Train set
train_metrics = {
    "Accuracy": accuracy_score(y_train, y_train_pred),
    "Precision": precision_score(y_train, y_train_pred),
    "Recall": recall_score(y_train, y_train_pred),
    "F1 Score": f1_score(y_train, y_train_pred)
}


# In[85]:


# Calculate metrics for Test set
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1 Score": f1_score(y_test, y_test_pred)
}


# In[86]:


train_metrics


# In[87]:


test_metrics


# In[88]:


import joblib
joblib.dump(best_log_reg, 'depression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and scaler saved.")


# In[ ]:




