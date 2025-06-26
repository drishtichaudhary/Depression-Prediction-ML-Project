Depression Prediction ML Project 🧠
Overview 🌟This project is a machine learning app built with Streamlit to predict depression likelihood from user inputs. It uses a Logistic Regression model trained on a dataset with features like Age, Work Pressure, Job Satisfaction, Sleep Duration, Work/Study Hours, and Financial Stress. SMOTE handles class imbalance, and it’s live on Streamlit Cloud! 🚀

Installation 🛠️Get this app running locally with these steps:  

Clone the Repo 📂:  
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Set Up a Virtual Environment (optional 😎):  
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies 📦:  
pip install -r requirements.txt

What’s in requirements.txt? 👇  
streamlit
numpy
scikit-learn
joblib
pandas
imbalanced-learn
seaborn
matplotlib
xgboost
⚠️App is built on Python - 3.10 environment⚠️

Launch the App 🎈:  
streamlit run app.py



Usage 🎮  

Fire up the app in your browser (locally at http://localhost:8501 or the deployed link) 🌐.  
Input these features:  
Age: 12–60 📏  
Work Pressure: 1–10 💼  
Job Satisfaction: 1–10 😊  
Sleep Duration: <5, 5-6, 6-7, 7-8, 8+ hours 😴  
Work/Study Hours: 0–18 hours ⏰  
Financial Stress: Yes/No 💸


Smash the "Predict" button to see: "Likely Depressed 😟" or "Not Depressed 🙂".

Dataset 📊The dataset (final_depression_dataset_1.csv) covers mental health and lifestyle features, with Depression (Yes/No) as the target. Key prep steps:  

Filled missing values with median (numerical) or mode (categorical) 🧹.  
Encoded Sleep Duration to 0–4 🔢.  
Used SMOTE to balance classes ⚖️.

Note: Dataset not in repo due to size/privacy. Hit up the repo owner for access 📧.  
Model 🤖  

Algorithm: Logistic Regression (L1 penalty, C=1, solver=liblinear) ⚙️.  
Preprocessing: SMOTE (sampling_strategy=0.5) and StandardScaler 📈.  
Features:  
Age  
Work Pressure  
Job Satisfaction  
Sleep Duration  
Work/Study Hours  
Financial Stress


Performance (test set) 📊:  
Accuracy: ~92.19% ✅  
Precision: ~77.42% 🎯  
Recall: ~79.12% 🔍  
F1 Score: ~78.26% 🌟



Saved as depression_model.pkl and scaler.pkl 💾.  
Features ✨  

Streamlit UI for easy input 🖱️.  
Real-time depression predictions with 6 features 🔮.  
Visuals like class distribution and histograms in Depression Prediction ML Project.py 📉.  
Compares Logistic Regression, Random Forest, and XGBoost models 🆚.

Deployment ☁️Hosted on Streamlit Cloud:  

URL: [https://depression-prediction-ml-project-vkxyr7ideplmnhaduwjvss.streamlit.app/] 🌍  
Push changes to GitHub, and Streamlit Cloud auto-redeploys 🔄.

Contributing 🤝  

Fork the repo 🍴.  
Create a branch (git checkout -b feature-branch) 🌿.  
Commit changes (git commit -m "Add feature") ✍️.  
Push (git push origin feature-branch) 🚀.  
Open a pull request 📬.

Follow PEP 8 and add comments for clarity 📝.  
