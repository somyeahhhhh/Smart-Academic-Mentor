An AI-powered student performance analysis system that predicts pass/fail outcomes and provides personalised recommendations using Machine Learning.

Project Files
File	Description
streamlit_app.py	Interactive faculty dashboard (Streamlit web app)
aiml.ipynb	Jupyter notebook — full ML pipeline with analysis
student_data.csv	Synthetic dataset — 300 students, 13 features
requirements.txt	All Python dependencies with pinned versions
project_report_aiml_updated.html	Full project report (open in any browser)
Setup — Run Once
Requires Python 3.11 (not 3.12, 3.13, or 3.14) Download: https://www.python.org/downloads/release/python-3119/ During install — tick "Add Python to PATH"

Install all dependencies:

py -3.11 -m pip install -r requirements.txt

How to Run
Streamlit Dashboard (recommended)
py -3.11 -m streamlit run streamlit_app.py

Opens at http://localhost:8501 in your browser.

Use the left sidebar to enter a student profile — the prediction, cluster map, and recommendations update instantly.

Jupyter Notebook
py -3.11 -m jupyter notebook aiml.ipynb

Opens in your browser. Click Cell → Run All to execute the full pipeline.

View HTML Report
Double-click project_report_aiml_updated.html — opens directly in any browser. No Python needed.

Models Used
Model	CV Accuracy	Notes
SVM (RBF)	82.99% ± 1.32%	Deployed model
Random Forest	~81%	Ensemble baseline
Logistic Regression	~78%	Linear baseline
Decision Tree	~75%	Interpretable baseline
Dataset
300 synthetic students generated to reflect realistic B.Tech distributions
Features: Attendance, Study Hours, Internal Marks, Assignment Submission, Previous CGPA, Subject Scores (Maths, Physics, Chemistry, Programming, English), Extracurricular Activity
Target: Final_Result (Pass / Fail)
Tech Stack
Python 3.11
scikit-learn (SVM, KMeans, PCA, cross-validation)
pandas, numpy
matplotlib, seaborn
Streamlit
Jupyter
