# 🏦 Loan Prediction Project

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python\&logoColor=white)]()
[![ML Model](https://img.shields.io/badge/Model-Custom%20Classifier-green)]()
[![Deployment](https://img.shields.io/badge/Deployment-Live-success?logo=streamlit)]()

---

## 🔗 Live Web App

⚡ Try it now:
[Loan Prediction App](https://loanpredictionproject-tejasgholap.streamlit.app/)

---

## 🧠 Project Summary

This web application predicts the **loan approval decision** for applicants based on financial, demographic, and credit-related inputs using machine learning. It is built to provide a seamless, friendly interface for both users and stakeholders.

---

## 🎯 Key Features

* Intuitive user input form with relevant variables
* Real-time prediction for loan approval (✅ Approved / ❌ Rejected)
* Pre-trained model, scaler and encoders loaded from a single Pickle file
* Responsive Streamlit UI with clean design
* Sidebar containing developer profile and links

---

## 📊 Input Variables

| Category          | Input Variables                                       |
| ----------------- | ----------------------------------------------------- |
| Applicant Profile | Gender, Married, Dependents, Education, Self-Employed |
| Financial Details | Applicant Income, Co-applicant Income                 |
| Loan Information  | Loan Amount, Loan Term (Months), Credit History       |
| Property Location | Property Area (Urban / Semiurban / Rural)             |

---

## 🧷 Technical Stack

| Layer            | Technologies & Libraries             |
| ---------------- | ------------------------------------ |
| User Interface   | Streamlit                            |
| Backend Logic    | Python                               |
| Machine Learning | scikit-learn, XGBoost, NumPy, Pandas |
| Model Storage    | Pickle (.pkl)                        |

---

## 📁 Project Directory

```
Loan-Prediction-Project/
│
├── streamlit_app.py           # Main Streamlit application
├── loan_approval_model.pkl    # Trained classifier + scaler + encoders
├── requirements.txt           # Library dependencies
└── README.md                  # Project documentation
```

---

## 🏁 Running Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/tejasgholap45/Loan-Prediction-Project.git
   cd Loan-Prediction-Project
   ```
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```
4. Open your browser and navigate to `http://localhost:8501`.

---

## 📦 requirements.txt

```
streamlit
pandas
numpy
scikit-learn
xgboost
pickle5
```

## 👤 Developer

**Tejas Gholap**
MCA Student | Machine Learning & AI Enthusiast

🔗 GitHub: [github.com/tejasgholap45](https://github.com/tejasgholap45)
🔗 LinkedIn: [linkedin.com/in/tejas-gholap-bb3417300/](https://www.linkedin.com/in/tejas-gholap-bb3417300/)
✉️ Email: [tejasgholap45@gmail.com](mailto:tejasgholap45@gmail.com)

---

## 🙏 Acknowledgements

* The open-source Streamlit community for providing easy deployment
* The authors and maintainers of scikit-learn and XGBoost
* Any dataset contributors or mentors who supported this project

---
