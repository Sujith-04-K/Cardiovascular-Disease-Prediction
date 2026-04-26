import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from utils import feature_engineering

def train():
    df = pd.read_csv("data/cardio_train.csv", sep=';')

    df = feature_engineering(df)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X = df.drop('cardio', axis=1)
    y = df['cardio']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save model + scaler
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("[INFO] Model saved successfully!")

if __name__ == "__main__":
    train()