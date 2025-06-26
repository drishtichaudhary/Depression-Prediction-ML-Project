#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Define the file path
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Projects\final_depression_dataset_1.csv"


# In[3]:


# Load the dataset
df = pd.read_csv(file_path)


# In[4]:


# Display basic info
print(df.info())
print(df.head())  # Show first 5 rows


# In[5]:


# Check missing values
print(df.isnull().sum())


# In[6]:


# Drop columns with too many missing values
df.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction'], inplace=True)


# In[7]:
# Fill numerical missing values with median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Work Pressure'] = df['Work Pressure'].fillna(df['Work Pressure'].median())
df['Job Satisfaction'] = df['Job Satisfaction'].fillna(df['Job Satisfaction'].median())
df['Work/Study Hours'] = df['Work/Study Hours'].fillna(df['Work/Study Hours'].median())
df['Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].mode()[0] if not df['Financial Stress'].mode().empty else 0)
# Sleep Duration handled in Cell [11]
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])  # Keep for dataset integrity
# In[9]:


# Verify if all missing values are handled
print(df.isnull().sum())  


# In[10]:
from sklearn.preprocessing import LabelEncoder

# Check unique values in Depression to debug
print("Unique values in Depression before encoding:", df['Depression'].unique())

# Fill NaNs in Depression with mode (or default 'No' if mode is empty)
df['Depression'] = df['Depression'].fillna(df['Depression'].mode()[0] if not df['Depression'].mode().empty else 'No')

# Convert to string and standardize case to handle inconsistencies (e.g., 'yes', 'YES')
df['Depression'] = df['Depression'].astype(str).str.strip().str.lower()
df['Depression'] = df['Depression'].replace({'yes': 'Yes', 'no': 'No'})

# Binary Encoding
binary_cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Depression']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    # Fill any remaining NaNs (from unmapped values) with mode or default
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

# Verify no NaNs in Depression
print("NaNs in Depression after encoding:", df['Depression'].isnull().sum())

# In[11]:

# Manual encoding for Sleep Duration to match app.py
print("Unique values in Sleep Duration before encoding:", df['Sleep Duration'].unique())
sleep_dict = {"<5": 0, "5-6": 1, "6-7": 2, "7-8": 3, "8+": 4}
df['Sleep Duration'] = df['Sleep Duration'].astype(str).str.strip()  # Convert to string and clean
df['Sleep Duration'] = df['Sleep Duration'].fillna('<5')  # Default to '<5'
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_dict)
df['Sleep Duration'] = df['Sleep Duration'].fillna(0)  # Fallback to 0 if mapping fails
print("NaNs in Sleep Duration after encoding:", df['Sleep Duration'].isnull().sum())
# In[12]:


# Skip one-hot encoding since app.py doesn't use City, Profession, Degree
# Label Encoding	When the categorical variable has an order/rank (e.g., Small < Medium < Large).
#One-Hot Encoding	When the categorical variable has no inherent order (e.g., City, Country, Profession).


# In[13]:


# Check dataset after encoding
print(df.head())  


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Countplot to visualize class imbalance
sns.countplot(x=df['Depression'])
plt.title("Class Distribution in 'Depression' Column")
plt.show()


# In[15]:


# Print value counts
print(df['Depression'].value_counts())


# In[16]:


#SMOTE (Synthetic Minority Oversampling Technique) creates synthetic examples for the minority class instead of just duplicating them.
from imblearn.over_sampling import SMOTE

# Define SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Adjust ratio as needed


# In[20]:


# Separate features and target
X_final = df[['Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
              'Work/Study Hours', 'Financial Stress']]
y_final = df['Depression']
# In[21]:


# Apply SMOTE
from imblearn.over_sampling import SMOTE

# Check for NaNs in X_final
print("NaNs in X_final:", X_final.isnull().sum())
if X_final.isnull().any().any():
    raise ValueError("NaNs found in X_final. Check missing value handling.")

# In[22]:


smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_final, y_final)


# In[23]:


# Convert back to DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_final.columns), pd.DataFrame(y_resampled, columns=['Depression'])], axis=1)


# In[24]:


# Check new class distribution
print(df_resampled['Depression'].value_counts())


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot class distribution
sns.countplot(x=y_resampled, palette='coolwarm')
plt.title("Class Distribution of Depression (After Balancing)")
plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


# In[26]:


# Plot histograms for all numerical features
X_resampled.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions (Histograms)", fontsize=16)
plt.show()


# In[28]:


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


# In[29]:


# Compute the correlation matrix
corr_matrix = X_resampled.corr()


# In[30]:


# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[31]:


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


# In[32]:


# Detect outliers
outliers_iqr = detect_outliers_iqr(X_resampled)
print("Outlier counts per feature (IQR method):")
print(outliers_iqr)


# In[35]:


# Drop 'Name' and 'City' (if still present)
X_resampled = X_resampled.drop(columns=['Name'], errors='ignore')


# In[37]:


# Check the remaining features
print("Remaining Features:", X_resampled.columns)


# In[38]:


from sklearn.preprocessing import StandardScaler

# Selecting numerical features
num_features = ['Age', 'Work/Study Hours', 'Financial Stress',
                'Sleep Duration', 'Job Satisfaction', 'Work Pressure']

# In[39]:

scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final)
# In[40]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Initialize Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)


# In[41]:


# Get feature importance
feature_importances = pd.Series(rf.feature_importances_, index=X_final.columns)
feature_importances.sort_values(ascending=False, inplace=True)


# In[42]:


# Display top features
print(feature_importances.head(10))


# In[43]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_final_scaled, y_final, 
                                                    test_size=0.2, random_state=42, 
                                                    stratify=y_final)
# In[44]:


# Standardize numerical features
# No need to scale again since X_final is already scaled
X_train_scaled = X_train
X_test_scaled = X_test

# In[45]:


# Train Logistic Regression model
best_log_reg = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42)
best_log_reg.fit(X_train_scaled, y_train)


# In[46]:


# Predict on test data
y_pred = best_log_reg.predict(X_test_scaled)


# In[47]:


# Evaluate model performance
best_log_reg_results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred)
}


# In[48]:


best_log_reg_results


# In[49]:


# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[50]:


# Train the model
rf_model.fit(X_train, y_train)


# In[51]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)


# In[52]:


# Evaluate the model
rf_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1 Score': f1_score(y_test, y_pred_rf)
}


# In[54]:


rf_metrics


# In[56]:


# Make sure xgboost is installed: pip install xgboost


# In[57]:


from xgboost import XGBClassifier
# Initialize and train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


# In[58]:


# Make predictions
y_pred_xgb = xgb_model.predict(X_test)


# In[59]:


# Evaluate performance
xgb_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1 Score': f1_score(y_test, y_pred_xgb)
}


# In[60]:


xgb_metrics


# In[64]:


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


# In[65]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # L1 (Lasso), L2 (Ridge)
    'solver': ['liblinear']  # 'liblinear' supports L1 & L2 penalties
}


# In[66]:


# Initialize Logistic Regression
log_reg = LogisticRegression(random_state=42)


# In[67]:


# Perform Grid Search
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[68]:


# Best parameters
print("Best Parameters:", grid_search.best_params_)


# In[69]:


# Train the best model
best_log_reg = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42)
best_log_reg.fit(X_train, y_train)


# In[70]:


# Predictions
y_pred = best_log_reg.predict(X_test)


# In[72]:


from sklearn import metrics

optimized_metrics = {
    "Accuracy": metrics.accuracy_score(y_test, y_pred),
    "Precision": metrics.precision_score(y_test, y_pred),
    "Recall": metrics.recall_score(y_test, y_pred),
    "F1 Score": metrics.f1_score(y_test, y_pred)  # Avoids conflict
}


# In[73]:


optimized_metrics


# In[76]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get predictions for both train & test sets
y_train_pred = best_log_reg.predict(X_train)  # Predictions on train data
y_test_pred = best_log_reg.predict(X_test)    # Predictions on test data


# In[77]:


# Calculate metrics for Train set
train_metrics = {
    "Accuracy": accuracy_score(y_train, y_train_pred),
    "Precision": precision_score(y_train, y_train_pred),
    "Recall": recall_score(y_train, y_train_pred),
    "F1 Score": f1_score(y_train, y_train_pred)
}


# In[78]:


# Calculate metrics for Test set
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1 Score": f1_score(y_test, y_test_pred)
}


# In[80]:


train_metrics


# In[81]:


test_metrics


# In[82]:
import joblib
joblib.dump(best_log_reg, 'depression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')



