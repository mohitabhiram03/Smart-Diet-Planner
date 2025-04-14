import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt

# ### Define Model Classes

class CaloriesModel(nn.Module):
    def __init__(self):
        super(CaloriesModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 128),  # 7 input features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

class NutritionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(NutritionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# ### Train and Save Calories Prediction Model

# Load and preprocess data
calories_data = pd.read_csv('datasets/human_input_to_calories_dataset.csv')
label_encoder = LabelEncoder()
calories_data['gender'] = label_encoder.fit_transform(calories_data['gender'])
features = ['age', 'weight(kg)', 'height(m)', 'gender', 'BMI', 'BMR', 'activity_level']
target = 'calories_to_maintain_weight'
X = calories_data[features].values
y = calories_data[target].values

# Scale features and target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize and train the calories model
calories_model = CaloriesModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(calories_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

epochs = 20
for epoch in range(epochs):
    calories_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = calories_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f"Calories Model - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# ### Evaluate the Calories Model

calories_model.eval()
with torch.no_grad():
    test_outputs = calories_model(X_test_tensor)

# Inverse transform predictions and actual values to original scale
predicted_calories = (test_outputs.numpy() * y_std) + y_mean
actual_calories = (y_test_tensor.numpy() * y_std) + y_mean

# Calculate model performance metrics
mae = mean_absolute_error(actual_calories, predicted_calories)
mse = mean_squared_error(actual_calories, predicted_calories)
rmse = np.sqrt(mse)
r2 = r2_score(actual_calories, predicted_calories)
mape = np.mean(np.abs((actual_calories - predicted_calories) / actual_calories)) * 100

# --- Improved Baseline ---
# Train a linear regression baseline
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
baseline_pred_scaled = baseline_model.predict(X_test)

# Inverse scaling for baseline predictions
baseline_pred = (baseline_pred_scaled * y_std) + y_mean

# Calculate baseline metrics
baseline_mae = mean_absolute_error(actual_calories, baseline_pred)
baseline_mse = mean_squared_error(actual_calories, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(actual_calories, baseline_pred)
baseline_mape = np.mean(np.abs((actual_calories - baseline_pred) / actual_calories)) * 100

# Print performance metrics
print("\nCalories Model Performance:")
print(f"Model MAE: {mae:.2f} kcal")
print(f"Model MSE: {mse:.2f}")
print(f"Model RMSE: {rmse:.2f} kcal")
print(f"Model R²: {r2:.4f}")
print(f"Model MAPE: {mape:.2f}%")
print("\nImproved Baseline Performance:")
print(f"Baseline MAE: {baseline_mae:.2f} kcal")
print(f"Baseline MSE: {baseline_mse:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f} kcal")
print(f"Baseline R²: {baseline_r2:.4f}")
print(f"Baseline MAPE: {baseline_mape:.2f}%")

# Visualize predicted vs actual calories
plt.figure(figsize=(8, 6))
plt.scatter(actual_calories, predicted_calories, alpha=0.5, label='Model Predictions')
plt.scatter(actual_calories, baseline_pred, alpha=0.5, label='Baseline Predictions')
plt.plot([min(actual_calories), max(actual_calories)], 
         [min(actual_calories), max(actual_calories)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Calories (kcal)')
plt.ylabel('Predicted Calories (kcal)')
plt.title('Predicted vs Actual Calories')
plt.legend()
plt.show()

# Save calories model and related objects
torch.save(calories_model.state_dict(), 'calories_model.pth')
joblib.dump(scaler_X, 'calories_scaler_X.pkl')
joblib.dump(label_encoder, 'gender_label_encoder.pkl')
joblib.dump({'mean': y_mean, 'std': y_std}, 'calories_y_scaler_params.pkl')

# ### Train and Save Nutrient Prediction Models for Each Meal

meals = ['breakfast', 'lunch', 'dinner']
for meal in meals:
    data = pd.read_csv(f"{meal}_data.csv")
    features = ['Caloric Value']
    targets = [col for col in data.columns if col not in ['Caloric Value', 'type', 'food', "Unnamed: 0.1", "Unnamed: 0", "Cluster"]]
    X = data[features].values
    y = data[targets].values

    # Scale features and targets
    scaler_X_meal = StandardScaler()
    scaler_y_meal = StandardScaler()
    X = scaler_X_meal.fit_transform(X)
    y = scaler_y_meal.fit_transform(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize and train the nutrient model
    model = NutritionModel(input_size=1, output_size=len(targets))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"{meal.capitalize()} Nutrient Model - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    # Save nutrient model and related objects
    torch.save(model.state_dict(), f'{meal}_nutrient_model.pth')
    joblib.dump(scaler_X_meal, f'{meal}_scaler_X.pkl')
    joblib.dump(scaler_y_meal, f'{meal}_scaler_y.pkl')
    joblib.dump(targets, f'{meal}_targets.pkl')
    data.to_pickle(f'{meal}_data.pkl')

print("All models and datasets have been saved as .pkl files.")