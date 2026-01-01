import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib

NBA_SINGLE_GAME_TRAIN_DATA_PATH = '/Users/justinwebster/Documents/Software Development Files/NBA-MODEL/NBA DATA/NBA_SINGLE_GAME_TRAINING_DATA.csv'

def csv_to_df(csv):
    return pd.read_csv(csv)

df = csv_to_df(NBA_SINGLE_GAME_TRAIN_DATA_PATH)

y = df['away_DRTG']

X = df[['away_DRB','away_BLK','home_TOV%', 'home_FTA']]

# Add a constant (intercept) to the independent variables
X_data_with_const = sm.add_constant(X)

# Create a DataFrame to store VIF results
vif_df = pd.DataFrame()
vif_df["Feature"] = X_data_with_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_data_with_const.values, i) for i in range(X_data_with_const.shape[1])]

# Print the VIF results
print(vif_df)

# 1. Split data into training and validation sets FIRST
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Instantiate StandardScaler
scaler = StandardScaler()

# 3. Fit the scaler ONLY on the training data, then transform training data
X_train_scaled = scaler.fit_transform(X_train)

# 4. Transform the validation data using the scaler fitted on the training data
X_val_scaled = scaler.transform(X_val)


# Instantiate the SGDRegressor model
sgd_mae_ridge_model = SGDRegressor(loss='epsilon_insensitive',
                                   epsilon=0.0,
                                   penalty='l2',
                                   alpha=0.001,
                                   max_iter=1,
                                   eta0=0.07,
                                   random_state=42,
                                   warm_start=True)

# Define mini-batch size and number of epochs
batch_size = 6
n_epochs = 80

# Lists to store performance metrics for plotting
train_mae_history = []
val_mae_history = []
train_r2_history = []
val_r2_history = []

# Training loop with mini-batches and performance tracking
n_batches = int(np.ceil(len(X_train_scaled) / batch_size)) # Use scaled training data length

for epoch in range(n_epochs):
    # Shuffle the data at the beginning of each epoch
    permutation = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[permutation]
    y_shuffled = y_train.iloc[permutation]

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train_scaled))
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]

        # Use partial_fit to train on each mini-batch
        sgd_mae_ridge_model.partial_fit(X_batch, y_batch)

    # --- Record performance at the end of each epoch ---
    # Predict on the full training set (scaled)
    y_train_pred = sgd_mae_ridge_model.predict(X_train_scaled)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Predict on the validation set (scaled)
    y_val_pred = sgd_mae_ridge_model.predict(X_val_scaled)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    train_mae_history.append(train_mae)
    val_mae_history.append(val_mae)
    train_r2_history.append(train_r2)
    val_r2_history.append(val_r2)

    # Optional: Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{n_epochs} - Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}, Train R2: {train_r2:.3f}, Val R2: {val_r2:.3f}")

# --- Plotting the Learning Curves ---
epochs_range = range(1, n_epochs + 1)

plt.figure(figsize=(12, 5))

# Plot MAE Learning Curve
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_mae_history, label='Training MAE', color='blue')
plt.plot(epochs_range, val_mae_history, label='Validation MAE', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Learning Curve (MAE)')
plt.legend()
plt.grid(True)

# Plot R-squared Learning Curve
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_r2_history, label='Training R-squared', color='blue')
plt.plot(epochs_range, val_r2_history, label='Validation R-squared', color='orange')
plt.xlabel('Epochs')
plt.ylabel('R-squared')
plt.title('Learning Curve (R-squared)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final Evaluation on the test set (using X_val, y_val in this scenario)
final_y_pred = sgd_mae_ridge_model.predict(X_val_scaled) # Predict using scaled validation data
final_r2 = r2_score(y_val, final_y_pred)
final_mae = mean_absolute_error(y_val, final_y_pred)

print("\n--- Final Model Evaluation ---")
print(f"Final R-squared: {final_r2:.3f}")
print(f"Final MAE: {final_mae:.3f}")
print("Final Coefficients:", sgd_mae_ridge_model.coef_)
print("Final Intercept:", sgd_mae_ridge_model.intercept_)

# Save the trained model
#joblib.dump(sgd_mae_ridge_model, 'nba_sgd_model.joblib')

# Save the fitted scaler
#joblib.dump(scaler, 'nba_scaler.joblib')