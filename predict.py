import pandas as pd
import joblib
import argparse

def predict_weather(temp, humidity, wind, pressure):
    # Load the trained model
    try:
        model = joblib.load("weather_model.pkl")
    except FileNotFoundError:
        print("Model not found. Please run train_model.py first.")
        return
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'temperature_c': [temp],
        'humidity_percent': [humidity],
        'wind_speed_kmh': [wind],
        'pressure_hpa': [pressure]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    print("\n--- Weather Prediction ---")
    print(f"Current Temperature: {temp}°C")
    print(f"Current Humidity: {humidity}%")
    print(f"Current Wind Speed: {wind} km/h")
    print(f"Current Pressure: {pressure} hPa")
    print(f"=> Predicted Tomorrow's Temperature: {prediction:.2f}°C")
    print("--------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict tomorrow's temperature")
    parser.add_argument("--temp", type=float, default=20.0, help="Current temperature in °C")
    parser.add_argument("--humidity", type=float, default=65.0, help="Current humidity in %")
    parser.add_argument("--wind", type=float, default=15.0, help="Current wind speed in km/h")
    parser.add_argument("--pressure", type=float, default=1013.0, help="Current atmospheric pressure in hPa")
    
    args = parser.parse_args()
    predict_weather(args.temp, args.humidity, args.wind, args.pressure)
