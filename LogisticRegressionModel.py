from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd



NBA_SINGLE_GAME_TRAIN_DATA_PATH = '/Users/justinwebster/Desktop/NBA-MODEL/NBA DATA/NBA_SINGLE_GAME_TRAINING_DATA'

def csv_to_df(csv):
    return pd.read_csv(csv)

df = csv_to_df(NBA_SINGLE_GAME_TRAIN_DATA_PATH)

y = df['GAME_RESULT']

X = df[['away_DRB','home_DRB','home_TOV%','away_TOV%', 'away_FTA','home_FTA']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

