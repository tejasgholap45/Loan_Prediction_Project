# ============================================
# 💳 Loan Approval Prediction (Final Fast Version)
# ============================================

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================
# 1️⃣ Database Connection
# ============================================
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='8080',
    database='loan_db'
)

# ============================================
# 2️⃣ Load Data from MySQL
# ============================================
df_applicant = pd.read_sql("SELECT * FROM applicant_info", conn)
df_financial = pd.read_sql("SELECT * FROM financial_info", conn)
df_loan = pd.read_sql("SELECT * FROM loan_info", conn)
conn.close()

# ============================================
# 3️⃣ Merge DataFrames
# ============================================
df = df_applicant.merge(df_financial, on='Loan_ID', how='inner')
df = df.merge(df_loan, on='Loan_ID', how='inner')

print("✅ Data loaded successfully!")
print("Shape of merged data:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# ============================================
# 4️⃣ Handle Missing Values
# ============================================
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# ============================================
# 5️⃣ Encode Categorical Columns
# ============================================
encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ============================================
# 6️⃣ Feature Selection
# ============================================
X = df.drop(columns=["Loan_Status", "Loan_ID"])
y = df["Loan_Status"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# 7️⃣ Standardize Data
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 8️⃣ Train Multiple Models
# ============================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    results.append([name, acc, prec, rec, f1])
    print(f"\n🔹 {name} Performance:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# ============================================
# 9️⃣ Compare Model Performance
# ============================================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\n📊 Model Comparison:\n", results_df)

# Plot Comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("model_comparison.png")  # ✅ Save instead of show
plt.close()  # ✅ Prevent hang

# ============================================
# 🔟 Save Best Model (Random Forest)
# ============================================
best_model = models["Random Forest"]
output_path = r"C:\all projects\data analysis project\mahice learning project\machine learning\end_to_end_project_loan\loan_approval_model.pkl"

try:
    with open(output_path, 'wb') as f:
        pickle.dump((best_model, scaler, encoders), f)
    print(f"\n✅ Model saved successfully at:\n{output_path}")
    print("📁 File exists:", os.path.exists(output_path))
except Exception as e:
    print("❌ Error while saving model:", e)
