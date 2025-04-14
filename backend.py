import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


class CaloriesModel(nn.Module):
    def __init__(self): 
        super(CaloriesModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 128),
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

# Train and Save Calories Prediction Model
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure correct shape
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train calories model
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

# Save calories model and related objects
torch.save(calories_model.state_dict(), 'calories_model.pth')
joblib.dump(scaler_X, 'calories_scaler_X.pkl')
joblib.dump(label_encoder, 'gender_label_encoder.pkl')
joblib.dump({'mean': y_mean, 'std': y_std}, 'calories_y_scaler_params.pkl')

# Train and Save Nutrient Prediction Models for Each Meal
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train nutrient model
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