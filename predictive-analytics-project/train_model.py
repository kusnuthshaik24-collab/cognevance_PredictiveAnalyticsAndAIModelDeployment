# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

print("Step 1: Generating synthetic healthcare data...")
np.random.seed(42)
n_samples = 2000

# Features
age = np.random.randint(18, 65, size=n_samples)
bmi = np.random.uniform(18.5, 40.0, size=n_samples)
smoker = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# Target: Medical charges (with non-linear relationships and noise)
charges = (age * 250) + (bmi * 400) + (smoker * 15000) + np.random.normal(0, 1000, size=n_samples)

df = pd.DataFrame({'age': age, 'bmi': bmi, 'smoker': smoker, 'charges': charges})

# Split Data
X = df[['age', 'bmi', 'smoker']]
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 2: Preprocessing and Feature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Step 3: Training Gradient Boosting Model...")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
train_r2 = model.score(X_train_scaled, y_train)
print(f"Model trained successfully. Train R² Score: {train_r2:.4f}")

print("Step 4: Saving models locally...")
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved 'best_model.pkl' and 'scaler.pkl' successfully!")