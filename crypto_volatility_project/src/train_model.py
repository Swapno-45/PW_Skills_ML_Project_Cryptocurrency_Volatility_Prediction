
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("data/crypto_data.csv")

# Feature Engineering
df['volatility'] = (df['high'] - df['low']) / df['open']
df['returns'] = df['close'].pct_change()
df['rolling_volatility'] = df['returns'].rolling(window=7).std()

df.dropna(inplace=True)

features = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
target = 'volatility'

X = df[features]
y = df[target]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

joblib.dump(model, "models/volatility_model.pkl")
