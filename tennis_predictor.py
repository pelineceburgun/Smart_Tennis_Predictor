import requests
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

def predict_court_availability(day_of_week: int, hour: int) -> int:
    if day_of_week in [5, 6] and 11 <= hour <= 14:
        return 1
    if (day_of_week == 4 or day_of_week == 5) and 18 <= hour <= 21:
        return 1
    if day_of_week in [0, 1, 2, 3, 4] and 11 <= hour <= 14:
        return 0
    if day_of_week in [0, 1, 2, 3, 6] and 18 <= hour <= 21:
        return 0
    return 0

def get_weather_data(city_name: str, api_key: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        print("Couldn't get weather data:", response.json())
        return None
    data = response.json()
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed'] * 3.6  # m/s → km/h
    return temperature, humidity, wind_speed

def predict_play_tennis(city: str, hour: int, api_key: str, model_path: str = 'logistic_model.pkl'):
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import pickle
    from tennis_predictor import predict_court_availability, get_weather_data  

    now = datetime.now()
    day_of_week = now.weekday()

    weather = get_weather_data(city, api_key)
    if not weather:
        return "Couldn't make prediction due to missing weather data."

    temperature, humidity, wind_speed = weather
    is_windy = 1 if wind_speed > 20 else 0
    court_availability = predict_court_availability(day_of_week, hour)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    day_vector = [0] * 7
    day_vector[day_of_week] = 1

    
    feature_names = ['temperature', 'humidity', 'wind_speed', 'court_availability', 'is_windy',
                     'hour_sin', 'hour_cos', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6']

   
    input_data = [temperature, humidity, wind_speed, court_availability, is_windy, hour_sin, hour_cos] + day_vector
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Load model and scaler
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    
    numeric_cols = ['temperature', 'humidity', 'wind_speed']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    
    prediction = model.predict(input_df)

    return " You should play tennis today!" if prediction[0] == 1 else " You shouldn't play tennis today."




