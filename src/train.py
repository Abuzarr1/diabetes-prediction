import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

def main():
    print("🚀 Starting Diabetes ML Training Pipeline...")

    # 1. Load Data
    data_path = 'data/Pakistani_Diabetes_Dataset.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    print(f"📁 Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features.")

    # 2. Preprocessing
    # Separate target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # One-hot encode categorical columns (Region)
    X = pd.get_dummies(X, columns=['Region'], drop_first=False)
    
    # Save the exact column order so the inference pipeline (dashboard) can match it
    model_columns = list(X.columns)
    
    # 3. Train/Test Split
    print("✂️ Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling
    # We only scale the numeric features (not the one-hot encoded regions or binary features)
    # But for simplicity and to match the original notebook, we'll scale everything
    # (StandardScaler handles 0/1 fine, though it's mathematically unnecessary, it's standard practice)
    print("⚖️ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Model
    print("🏋️ Training Random Forest Classifier...")
    # Class weights to handle any minor imbalance
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    print("✅ Model fitting complete.")

    # 6. Evaluation
    print("📊 Evaluating model performance on test set...")
    y_pred = rf_model.predict(X_test_scaled)
    y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- 🏆 Model Performance Summary ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print("-------------------------------------\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # 7. Serialization
    print("💾 Saving artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(rf_model, 'models/diabetes_rf_model.pkl')
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    # Save the column order
    joblib.dump(model_columns, 'models/model_columns.pkl')

    print("✅ Pipeline complete! Model and preprocessors saved to models/ directory.")

if __name__ == "__main__":
    main()
