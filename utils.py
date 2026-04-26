import pandas as pd

def feature_engineering(df):
    print("[INFO] Performing feature engineering...")

    # Age groups
    df['age_years'] = df['age'] // 365
    df['age_group'] = pd.cut(df['age_years'], bins=[0,30,50,70,100], labels=[0,1,2,3])

    # BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Blood pressure difference
    df['bp_diff'] = df['ap_hi'] - df['ap_lo']

    # Risk flags
    df['high_bp'] = (df['ap_hi'] > 140).astype(int)
    df['high_chol'] = (df['cholesterol'] > 1).astype(int)

    return df