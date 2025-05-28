import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def run_preprocessing(input_path=r'D:\Customer Churn\data\WA_Fn-UseC_-Telco-Customer-Churn.csv',
                      output_path='data/preprocessed_churn_data.csv',
                      scaler_path='models/preprocessor.joblib',
                      output_with_id_path='data/preprocessed_churn_with_id.csv'):
    # Load raw dataset
    df = pd.read_csv(input_path)
    customer_ids = df['customerID'].values  # Save customer IDs separately

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Handle missing TotalCharges (where tenure == 0)
    df['TotalCharges'].fillna(0, inplace=True)

    # Separate numerical and categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Handle 'SeniorCitizen' as categorical
    if 'SeniorCitizen' in numerical_cols:
        numerical_cols.remove('SeniorCitizen')
        categorical_cols.append('SeniorCitizen')

    # Drop customerID and derived columns
    df.drop(columns=['customerID'], inplace=True)
    categorical_cols.remove('customerID')

    # Normalize "No internet service" to "No" in internet-related columns
    internet_related_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in internet_related_cols:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')

    # Split target
    target = df['Churn']
    df.drop(columns='Churn', inplace=True)

    # Scale numerical columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

    # One-hot encode categorical columns (excluding 'Churn')
    categorical_cols_no_target = [col for col in categorical_cols if col != 'Churn']
    df_encoded = pd.get_dummies(df[categorical_cols_no_target], drop_first=True).astype(int)

    # Combine
    df_final = pd.concat([df_scaled, df_encoded, target], axis=1)

    # Save processed dataset for training (no ID)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

    # Save version with ID for Streamlit
    df_final_with_id = df_final.copy()
    df_final_with_id.insert(0, 'customerID', customer_ids)
    df_final_with_id.to_csv(output_with_id_path, index=False)

    # Save the scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print(" Preprocessing complete.")
    print("Training CSV saved to:", output_path)
    print("Streamlit CSV (with ID) saved to:", output_with_id_path)

if __name__ == "__main__":
    run_preprocessing()
