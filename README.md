# 🧠 Depression Prediction ML Project

A Streamlit-powered web app that predicts depression likelihood from lifestyle and mental health factors using a Logistic Regression model. It works on a Logistic Regression model trained on a dataset with features like Age, Work Pressure, Job Satisfaction, Sleep Duration, Work/Study Hours, and Financial Stress. Built with care, scaled with SMOTE, and deployed for everyone. 🎯

---

## 🌟 Overview

This machine learning project predicts whether a person is likely to be depressed based on key features like:

- Age
- Work Pressure
- Job Satisfaction
- Sleep Duration
- Work/Study Hours
- Financial Stress

Key highlights:
- **Balanced dataset** using **SMOTE**
- **Logistic Regression model** (L1 penalty, `C=1`)
- **Deployed live on Streamlit Cloud!**

---

## 🛠️ Installation

Run the app locally by following these steps:

### 🔁 Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
###🧪 (Optional) Set Up a Virtual Environment
```bash
python -m venv venv
```

### 📦 Install Dependencies
```
pip install -r requirements.txt

<details><summary>📋<b>Dependencies in requirements.txt</b></summary>
```
streamlit
numpy
scikit-learn
joblib
pandas
imbalanced-learn
seaborn
matplotlib
xgboost
⚠️ Built on Python 3.10
```
</details>
```
### 🚀 Launch the App
```bash
streamlit run app.py
```

## 🎮 Usage

Open the app at http://localhost:8501 or use the deployed link (see below). 

👉 Input Fields: 
Age: 12–60 

Work Pressure: 1–10 

Job Satisfaction: 1–10 

Sleep Duration: <5, 5–6, 6–7, 7–8, 8+ hours 

Work/Study Hours: 0–18 hours 

Financial Stress: Yes/No 

Click "Predict" to get: 

"Likely Depressed 😟" or "Not Depressed 🙂" 

## 📊 Dataset

The dataset (final_depression_dataset_1.csv) includes mental health and lifestyle factors. The target variable is Depression (Yes/No). 

Key Preprocessing Steps: 

Missing values filled (median for numeric, mode for categorical) 

Encoded categorical features 

SMOTE used to handle class imbalance (sampling_strategy=0.5) 

📌 Note: Dataset not included due to privacy. Contact the repo owner for access. 

## 🤖 Model Details

Algorithm: Logistic Regression (L1 penalty, C=1, solver=liblinear)

Features Used:

Age

Work Pressure

Job Satisfaction

Sleep Duration

Work/Study Hours

Financial Stress

Preprocessing: StandardScaler + SMOTE

## 📈 Model Performance (Test Set)

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | 92.19% ✅  |
| Precision | 77.42% 🎯 |
| Recall    | 79.12% 🔍 |
| F1 Score  | 78.26% 🌟 |

✅ Saved model: depression_model.pkl 
✅ Saved scaler: scaler.pkl 

## ☁️ Deployment

🌍 Live App: 
https://depression-prediction-ml-project-vkxyr7ideplmnhaduwjvss.streamlit.app/ 

💡 Pushing changes to GitHub automatically updates the deployed version on Streamlit Cloud. 

## 📬 Contact

For dataset access or queries, please contact the repository owner via GitHub or email (drishtichaudhary616@gmail.com).
