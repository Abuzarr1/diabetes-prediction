import pandas as pd
import numpy as np

def generate_diabetes_data(n_samples=5000):
    np.random.seed(42)
    
    # Target variable
    # ~35% prevalence for realistic clinical dataset
    outcome = np.random.binomial(1, 0.35, n_samples)
    
    # Base features
    regions = ['Punjab', 'Sindh', 'KPK', 'Balochistan']
    region = np.random.choice(regions, n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # Correlated features based on outcome
    # Diabetic patients generally have higher weight, waist size, BP, and blood sugar
    
    weight = np.where(outcome == 1, 
                      np.random.normal(80, 15, n_samples),
                      np.random.normal(75, 12, n_samples))
    weight = np.clip(weight, 40, 150)
    
    waist = np.where(outcome == 1,
                     np.random.normal(38, 6, n_samples),
                     np.random.normal(34, 5, n_samples))
    waist = np.clip(waist, 24, 60)
    
    sys_bp = np.where(outcome == 1,
                      np.random.normal(135, 15, n_samples),
                      np.random.normal(125, 15, n_samples))
    sys_bp = np.clip(sys_bp, 90, 200)
    
    dia_bp = np.where(outcome == 1,
                      np.random.normal(90, 12, n_samples),
                      np.random.normal(80, 10, n_samples))
    dia_bp = np.clip(dia_bp, 60, 120)
    
    family_hist = np.where(outcome == 1,
                           np.random.binomial(1, 0.6, n_samples),
                           np.random.binomial(1, 0.2, n_samples))
    
    # Clinical markers
    hba1c = np.where(outcome == 1,
                     np.random.normal(7.5, 1.2, n_samples),
                     np.random.normal(5.5, 0.8, n_samples))
    hba1c = np.clip(hba1c, 4.0, 14.0)
    
    blood_sugar = np.where(outcome == 1,
                           np.random.normal(160, 40, n_samples),
                           np.random.normal(105, 20, n_samples))
    blood_sugar = np.clip(blood_sugar, 70, 400)
    
    # Symptoms
    vision = np.where(outcome == 1,
                      np.random.binomial(1, 0.4, n_samples),
                      np.random.binomial(1, 0.05, n_samples))
                      
    exercise = np.where(outcome == 1,
                        np.random.gamma(2, 20, n_samples),
                        np.random.gamma(4, 30, n_samples))
    exercise = np.clip(exercise, 0, 400).astype(int)
    
    polydipsia = np.where(outcome == 1,
                          np.random.binomial(1, 0.5, n_samples),
                          np.random.binomial(1, 0.02, n_samples))
                          
    polyuria = np.where(outcome == 1,
                        np.random.binomial(1, 0.55, n_samples),
                        np.random.binomial(1, 0.01, n_samples))
                        
    # Only diabetics have diabetes duration (mostly) or we just leave it for all (as 0 for non-diabetics)
    duration = np.where(outcome == 1,
                        np.random.gamma(2, 3, n_samples),
                        np.zeros(n_samples))
    duration = np.clip(duration, 0, 30).astype(int)
    
    nephropathy = np.where(outcome == 1,
                           np.random.binomial(1, 0.2, n_samples),
                           np.zeros(n_samples)) # Very rare in non-diabetics

    df = pd.DataFrame({
        'Region': region,
        'Weight': np.round(weight, 1),
        'WaistSize': np.round(waist, 1),
        'SysBP': np.round(sys_bp, 0).astype(int),
        'DiaBP': np.round(dia_bp, 0).astype(int),
        'FamilyHistory': family_hist,
        'HbA1c': np.round(hba1c, 1),
        'BloodSugar': np.round(blood_sugar, 0).astype(int),
        'VisionIssue': vision,
        'Exercise': exercise,
        'Polydipsia': polydipsia,
        'Polyuria': polyuria,
        'DiabetesDuration': duration,
        'Nephropathy': nephropathy.astype(int),
        'Outcome': outcome
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic diabetes dataset...")
    df = generate_diabetes_data(5000)
    df.to_csv('data/Pakistani_Diabetes_Dataset.csv', index=False)
    print(f"Generated dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Saved to data/Pakistani_Diabetes_Dataset.csv")
