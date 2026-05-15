import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train():
    print("Loading data...")
    df = pd.read_csv("weather_data.csv")
    
    # Feature columns: date, temperature_c, humidity_percent, wind_speed_kmh, pressure_hpa
    # We will exclude 'date' for this simple model
    features = ['temperature_c', 'humidity_percent', 'wind_speed_kmh', 'pressure_hpa']
    X = df[features]
    y = df['target_temp_c']
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Absolute Error: {mae:.2f}°C")
    print(f"R² Score: {r2:.2f}")
    
    # Save the model
    model_path = "weather_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
