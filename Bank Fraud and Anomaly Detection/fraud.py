

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv('C:/Users/HP/Downloads/bank_fraud/Fraud Detection Transactions Dataset.csv')

df = df.drop(["Transaction_ID", "User_ID", "Timestamp"], axis=1)

df = df.drop("Risk_Score", axis=1)

df = df.fillna(0)


y = df['Fraud_Label']
X = df.drop('Fraud_Label', axis=1)


X = pd.get_dummies(X, drop_first=True)

training_columns = X.columns



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print("===== EXISTING DATA PREDICTIONS =====")
print("First 10 Predictions:", y_pred[:10])
print("First 10 Actual Values:", y_test.values[:10])

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))



new_transaction_raw = pd.DataFrame([{
    'Transaction_Amount': float(input("Transaction Amount: ")),
    'Account_Balance': float(input("Account Balance: ")),
    'IP_Address_Flag': int(input("IP Address Flag (0 or 1): ")),
    'Previous_Fraudulent_Activity': int(input("Previous Fraud Activity (0 or 1): ")),
    'Daily_Transaction_Count': int(input("Daily Transaction Count: ")),
    'Avg_Transaction_Amount_7d': float(input("Avg Transaction Amount (7 days): ")),
    'Failed_Transaction_Count_7d': int(input("Failed Transaction Count (7 days): ")),
    'Card_Age': int(input("Card Age (days): ")),
    'Transaction_Distance': float(input("Transaction Distance (km): ")),
    'Is_Weekend': int(input("Is Weekend? (0 or 1): ")),
    'Transaction_Type': input("Transaction Type (Online/POS/ATM): "),
    'Device_Type': input("Device Type (Mobile/Desktop): "),
    'Location': input("Location: "),
    'Merchant_Category': input("Merchant Category: "),
    'Card_Type': input("Card Type (Visa/Mastercard): "),
    'Authentication_Method': input("Authentication Method (OTP/PIN/Biometric): ")
}])

new_encoded = pd.get_dummies(new_transaction_raw)

new_encoded = new_encoded.reindex(columns=training_columns, fill_value=0)



prediction = model.predict(new_encoded)
probability = model.predict_proba(new_encoded)

print("\nPredicted Fraud Label:", prediction[0])
print("Fraud Probability:", round(probability[0][1], 3))

if prediction[0] == 1:
    print(" Fraudulent transaction detected")
else:
    print(" Transaction is legitimate")








import joblib

joblib.dump(model, "fraud_model.pkl")
joblib.dump(training_columns, "training_columns.pkl")