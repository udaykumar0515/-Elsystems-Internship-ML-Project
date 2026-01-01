# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv(r'D:\uday\projects\Elsysytems\Customer-Churn.csv')

# Convert categorical columns using get_dummies
X = pd.get_dummies(data[['tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod']], drop_first=True)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(data['Churn'])  # Ensure 'Churn' matches the column name in your data

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model using pickle
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")
