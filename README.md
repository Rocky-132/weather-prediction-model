# weather-prediction-model
 Machine Learning Weather Prediction System built with Python & Scikit-learn. Generates synthetic weather data, trains a Random Forest model, visualizes trends, and predicts tomorrow’s temperature using humidity, pressure, wind speed, and temperature inputs. Includes data visualization and CLI support.

 Weather Prediction ML Project

 Overview

This project is a Machine Learning-based Weather Prediction System developed using Python and Scikit-learn. It predicts tomorrow’s temperature based on current weather conditions such as temperature, humidity, wind speed, and atmospheric pressure.

The project includes:

* Synthetic weather data generation
* Data visualization
* Machine Learning model training
* Weather prediction using trained model

---

 Features

* Generate synthetic weather datasets
* Train a Random Forest Regression model
* Predict future temperature values
* Visualize weather trends and model accuracy
* Simple command-line interface

---

Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

 Project Structure

```bash
├── generate_data.py      # Generate synthetic weather dataset
├── train_model.py        # Train ML model
├── predict.py            # Predict tomorrow's temperature
├── visualize.py          # Visualize data & predictions
├── weather_data.csv      # Generated dataset
├── weather_model.pkl     # Trained ML model
├── requirements.txt      # Required dependencies
```

---

 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/weather-prediction-ml.git
cd weather-prediction-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

 Usage

 Generate Dataset

```bash
python generate_data.py
```

 Train Model

```bash
python train_model.py
```

 Predict Weather

```bash
python predict.py --temp 30 --humidity 65 --wind 12 --pressure 1010
```

 Visualize Results

```bash
python visualize.py
```

---

 Model Details

The project uses:

* **Random Forest Regressor**
* Evaluation Metrics:

  * Mean Absolute Error (MAE)
  * R² Score

---

 Example Prediction

```bash
Current Temperature: 30°C
Current Humidity: 65%
Current Wind Speed: 12 km/h
Current Pressure: 1010 hPa

Predicted Tomorrow's Temperature: 29.42°C
```

---

 Future Improvements

* Use real-time weather APIs
* Add deep learning models (LSTM)
* Deploy using Flask or Streamlit
* Create interactive dashboard

---

 Author

Developed by Devendra Rakesh Vajrapu
