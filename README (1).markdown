# 🎾 Smart Tennis Predictor

> **"Is today a good day to play tennis?"**\
> This application uses a machine learning model to predict whether conditions are suitable for playing tennis based on factors such as temperature, humidity, wind conditions, time of day, and court occupancy.

---

## 🚀 Project Goal

The goal of this project is to predict the **suitability of playing tennis** based on weather conditions and court availability using machine learning.\
We aimed to make daily decision-making data-driven.

---

## 🧠 Features Used

The model makes predictions using the following features:

- 🌡️ `temperature`: Temperature (°C)
- 💧 `humidity`: Humidity (%)
- 🌬️ `wind_speed`: Wind speed (km/h)
- 🌪️ `is_windy`: 1 if wind speed &gt; 20 km/h, 0 otherwise
- 🕒 `hour_of_day`: Hour of the day (0-23), encoded using trigonometric functions (`sin`, `cos`)
- 📆 `day_of_week`: Day of the week (One-hot encoded)
- 🎾 `court_availability`: Estimated court occupancy based on hour and day (0 or 1)

---

## 🔍 Model Training Process

- **Data**: A synthetic dataset with 150 samples was generated using `numpy`.
- **Preprocessing**:
  - Numerical features: Standardized using `StandardScaler`
  - Time: Encoded with `sin` and `cos` functions
  - Categorical features: One-hot encoded using `pd.get_dummies`
- **Model Selection**: The following models were tested:
  - ✅ **Logistic Regression**
  - ✅ Decision Tree
  - ✅ Random Forest

### 🎯 Results:

| Model | Accuracy | Recall (class 1) | F1-score |
| --- | --- | --- | --- |
| **LogisticRegression** | **90%** | 75% | 80% |
| Decision Tree | 96% | 100% | 94% |
| Random Forest | 87% | 62% | 71% |

> Decision: The Logistic Regression model was chosen for its stable performance and **generalization capability**.

---

## 🌤️ Weather API Integration

Weather data is retrieved in real-time using the OpenWeatherMap API:

- Temperature, humidity, and wind information are fetched from the API
- Court availability: Estimated based on rules tied to hour and day

---

## 🖥️ Application Interface (Streamlit)

- Users input the city name and time
- The predicted result is displayed as text on the screen
- API key is managed through Streamlit secrets

---

![Placeholder for demo GIF](https://via.placeholder.com/600x300.png?text=Demo+Image+Coming+Soon)---

## 🛠️ Technologies Used

- `Python`
- `pandas`, `numpy`
- `scikit-learn`
- `requests` (API integration)
- `Streamlit` (UI)
- `pickle` (model serialization)