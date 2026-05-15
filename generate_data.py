import pandas as pd
import numpy as np
import os

def generate_weather_data(num_days=1000):
    print("Generating synthetic weather data...")
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start="2020-01-01", periods=num_days, freq="D")
    
    # Generate features
    # Temperature generally cycles yearly, but we'll add random noise
    day_of_year = dates.dayofyear
    base_temp = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365.25)
    temperature = base_temp + np.random.normal(0, 3, num_days)
    
    humidity = np.clip(np.random.normal(60, 15, num_days), 0, 100)
    wind_speed = np.clip(np.random.normal(10, 5, num_days), 0, None)
    pressure = np.random.normal(1013, 10, num_days)
    
    # Target: tomorrow's temperature
    # We will shift the temperature by -1
    
    df = pd.DataFrame({
        "date": dates,
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "wind_speed_kmh": wind_speed,
        "pressure_hpa": pressure
    })
    
    df['target_temp_c'] = df['temperature_c'].shift(-1)
    
    # Drop last row since it won't have a target
    df.dropna(inplace=True)
    
    # Save to CSV
    output_path = "weather_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows of weather data in {output_path}")

if __name__ == "__main__":
    generate_weather_data()
