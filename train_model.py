import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

np.random.seed(42)  

n_samples = 150

df = pd.DataFrame({
    'temperature': np.random.randint(15, 40, n_samples),
    'humidity': np.random.randint(20, 90, n_samples),
    'wind_speed': np.random.randint(0, 50, n_samples),
    'hour_of_day': np.random.randint(6, 22, n_samples),
    'court_availability': np.random.randint(0, 2, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
})

df['is_windy'] = (df['wind_speed'] > 30).astype(int)


df['will_play_tennis'] = (
    (df['is_windy'] == 0) &
    (df['court_availability'] == 1) &
    (df['temperature'] < 35)
).astype(int)


df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df.drop('hour_of_day', axis=1, inplace=True)
df = pd.get_dummies(df, columns=['day_of_week'], prefix='day')


X = df.drop('will_play_tennis', axis=1)
y = df['will_play_tennis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
numeric_cols = ['temperature', 'humidity', 'wind_speed']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model successfully saved.")
