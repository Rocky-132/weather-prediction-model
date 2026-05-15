import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

def plot_visualizations():
    print("Loading data and model...")
    try:
        df = pd.read_csv("weather_data.csv")
        model = joblib.load("weather_model.pkl")
    except FileNotFoundError:
        print("Required files not found. Please run generate_data.py and train_model.py first.")
        return
    
    features = ['temperature_c', 'humidity_percent', 'wind_speed_kmh', 'pressure_hpa']
    X = df[features]
    y = df['target_temp_c']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)
    
    # Set artistic style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Actual vs Predicted
    ax1.scatter(y_test, predictions, alpha=0.5, color='#3498db')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax1.set_title('Model Accuracy: Actual vs Predicted Temperature', fontsize=14)
    
    # Plot 2: Time Series of Temperature (First 150 days)
    df_subset = df.head(150)
    ax2.plot(pd.to_datetime(df_subset['date']), df_subset['temperature_c'], label='Temperature (°C)', color='#e74c3c')
    ax2.set_xlabel('Date (2020)', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title('Weather Data Trend (First 150 Days)', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    print("Opening visual charts window...")
    plt.show()

if __name__ == "__main__":
    plot_visualizations()
