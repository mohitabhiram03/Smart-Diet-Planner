import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import random

# Define model classes


class CaloriesModel(nn.Module):
    def __init__(self):
        super(CaloriesModel, self).__init__()
        self.fc = nn.Sequential(
            # 7 features: age, weight, height, gender, BMI, BMR, activity_level
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


# Load models and scalers
try:
    calories_model = CaloriesModel()
    calories_model.load_state_dict(torch.load('calories_model.pth'))
    calories_model.eval()

    calories_scaler_X = joblib.load('calories_scaler_X.pkl')
    gender_label_encoder = joblib.load('gender_label_encoder.pkl')
    y_scaler_params = joblib.load('calories_y_scaler_params.pkl')
    y_mean = y_scaler_params['mean']
    y_std = y_scaler_params['std']

    meals = ['breakfast', 'lunch', 'dinner']
    meal_models = {}
    meal_scalers_X = {}
    meal_scalers_y = {}
    meal_targets = {}
    meal_data = {}

    for meal in meals:
        targets = joblib.load(f'{meal}_targets.pkl')
        meal_models[meal] = NutritionModel(
            input_size=1, output_size=len(targets))
        meal_models[meal].load_state_dict(
            torch.load(f'{meal}_nutrient_model.pth'))
        meal_models[meal].eval()
        meal_scalers_X[meal] = joblib.load(f'{meal}_scaler_X.pkl')
        meal_scalers_y[meal] = joblib.load(f'{meal}_scaler_y.pkl')
        meal_targets[meal] = targets
        meal_data[meal] = pd.read_pickle(f'{meal}_data.pkl')
except Exception as e:
    st.error(f"Error loading models or datasets: {e}")
    st.stop()

# Define functions


def calculate_bmi(weight, height):
    if height <= 0:
        return None
    return weight / (height ** 2)


def calculate_bmr(age, weight, height, gender):
    height_cm = height * 100
    if gender == "M":
        return 88.362 + (13.397 * weight) + (4.799 * height_cm) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height_cm) - (4.330 * age)


def predict_calories(inputs):
    inputs_scaled = calories_scaler_X.transform([inputs])
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = calories_model(inputs_tensor).item()
    return prediction * y_std + y_mean


def adjust_calories(tdee, weight_goal):
    if weight_goal == "Maintain Weight":
        return tdee
    elif weight_goal == "Lose Weight":
        # Minimum calorie limits
        return max(tdee - 500, 1200 if gender == "F" else 1500)
    elif weight_goal == "Gain Weight":
        return tdee + 500


def divide_calories(total_calories):
    breakfast_ratio = 0.25
    lunch_ratio = 0.31
    dinner_ratio = 0.35
    breakfast_calories = total_calories * breakfast_ratio
    lunch_calories = total_calories * lunch_ratio
    dinner_calories = total_calories * dinner_ratio
    return breakfast_calories, lunch_calories, dinner_calories


def predict_nutrients(caloric_value, model, scaler_X, scaler_y):
    caloric_value_scaled = scaler_X.transform([[caloric_value]])
    caloric_value_tensor = torch.tensor(
        caloric_value_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(caloric_value_tensor).numpy()
    return scaler_y.inverse_transform(prediction)[0]


def select_meal_recipes(meal_df, target_calories, target_nutrients, max_attempts=10000, calorie_tolerance=0.15):
    """
    Select a combination of recipes for a meal that best matches the target calories and nutrients.

    Parameters:
    - meal_df: DataFrame with recipe data.
    - target_calories: Target caloric value for the meal.
    - target_nutrients: Dictionary of target nutrient values.
    - max_attempts: Maximum number of recipe combinations to try (default: 10000).
    - calorie_tolerance: Acceptable deviation from target calories as a fraction (default: 0.15, i.e., ±15%).

    Returns:
    - DataFrame with the selected recipes, or None if no recipes could be selected.
    """
    nutrient_priority = {
        'Protein': 0.15,
        'Carbohydrates': 0.15,
        'Fat': 0.15,
        'default': 0.25
    }

    def calculate_score(actual_cal, actual_nutrients, target_cal, target_nut):
        """
        Calculate a score for a combination based on how close it is to the target calories and nutrients.

        Parameters:
        - actual_cal: Total calories of the combination.
        - actual_nutrients: Dictionary of total nutrient values.
        - target_cal: Target caloric value.
        - target_nut: Dictionary of target nutrient values.

        Returns:
        - Average score (lower is better).
        """
        scores = []
        cal_error = abs(actual_cal - target_cal) / \
            target_cal if target_cal != 0 else (0 if actual_cal == 0 else 1)
        scores.append(min(cal_error / 0.10, 1.0))
        for nut, value in target_nut.items():
            actual = actual_nutrients[nut]
            tolerance = nutrient_priority.get(
                nut, nutrient_priority['default'])
            error = abs(actual - value) / \
                value if value != 0 else (0 if actual == 0 else 1)
            scores.append(min(error / tolerance, 1.0))
        return np.mean(scores)

    # Initialize variables to track the best combination within the calorie range
    best_score_in_range = float('inf')
    best_combination_in_range = None
    # Initialize variables to track the combination closest to the target calories
    smallest_cal_error = float('inf')
    best_combination_closest = None
    nutrient_cols = list(target_nutrients.keys())

    # Check for required columns in the dataset
    required_cols = ['Caloric Value'] + nutrient_cols
    missing_cols = [col for col in required_cols if col not in meal_df.columns]
    if missing_cols:
        st.error(f"Meal data missing columns: {missing_cols}")
        return None

    # Try different combinations of recipes
    for _ in range(max_attempts):
        num_recipes = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
        try:
            sample = meal_df.sample(n=num_recipes, replace=False)
        except ValueError:
            continue

        total_cal = sample['Caloric Value'].sum()
        # Check if the combination is within the acceptable calorie range
        if (target_calories * (1 - calorie_tolerance)) <= total_cal <= (target_calories * (1 + calorie_tolerance)):
            total_nutrients = sample[nutrient_cols].sum().to_dict()
            score = calculate_score(
                total_cal, total_nutrients, target_calories, target_nutrients)
            if score < best_score_in_range:
                best_score_in_range = score
                best_combination_in_range = sample

        # Track the combination closest to the target calories
        cal_error = abs(total_cal - target_calories)
        if cal_error < smallest_cal_error:
            smallest_cal_error = cal_error
            best_combination_closest = sample

    # If a combination within the range is found, return it
    if best_combination_in_range is not None:
        return best_combination_in_range
    else:
        # If no combination is within the range, return the closest one and warn the user
        st.warning(
            "No combination found within ±15% of the target calories. Selecting the closest available.")
        return best_combination_closest


def display_recipes(recipes, meal_name, target_calories, target_nutrients):
    if recipes is None or len(recipes) == 0:
        st.write(f"No {meal_name} recipes found")
        return

    # Convert the 'food' column to title case
    recipes['food'] = recipes['food'].str.title()

    st.subheader(f"{meal_name.upper()} ({len(recipes)} recipes)")
    st.write("*Selected Dishes:*", ", ".join(recipes['food'].tolist()))

    # Extract relevant columns for the table
    nutritional_cols = ['food', 'Caloric Value',
                        'Protein', 'Carbohydrates', 'Fat']
    nutritional_data = recipes[nutritional_cols]

    # Rename columns for better readability
    nutritional_data = nutritional_data.rename(columns={
        'food': 'Recipe',
        'Caloric Value': 'Calories (kcal)',
        'Protein': 'Protein (g)',
        'Carbohydrates': 'Carbs (g)',
        'Fat': 'Fat (g)'
    })

    # Remove the index by resetting it and dropping it
    nutritional_data = nutritional_data.reset_index(drop=True)

    # Display the table without the index and with formatted values
    st.write("**Nutritional Breakdown:**")
    st.dataframe(nutritional_data.style.format({
        'Calories (kcal)': '{:.1f}',
        'Protein (g)': '{:.1f}',
        'Carbs (g)': '{:.1f}',
        'Fat (g)': '{:.1f}'
        # Set hide_index=True to hide the index
    }), use_container_width=True, hide_index=True)

    # Calculate and display total calories and comparison to target
    total_cal = recipes['Caloric Value'].sum()
    st.write(f"**Total Calories:** {total_cal:.1f} kcal (target: {target_calories:.1f} kcal, "
             f"difference: {(total_cal - target_calories)/target_calories*100:.1f}%)")


# Streamlit app
st.title("Smart Diet Planner")

st.header("User Inputs")
age = st.number_input("Age", min_value=0, max_value=120, value=25)
weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
height = st.number_input("Height (m)", min_value=0.0, value=1.75)
gender = st.selectbox("Gender", ["M", "F"])
activity_level_str = st.selectbox(
    "Activity Level", ["Sedentary", "Active", "Very Active"])
weight_goal = st.selectbox(
    "Weight Goal", ["Maintain Weight", "Lose Weight", "Gain Weight"])

# Map activity level to numerical value
activity_level_map = {"Sedentary": 1.2, "Active": 1.55, "Very Active": 1.725}
activity_level = activity_level_map[activity_level_str]

# Input validation
if age < 5 or age > 90:
    st.error("Age must be between 5 and 90.")
elif weight <= 12 or weight >= 100:
    st.error("Weight must be between 12 kg and 100 kg.")
elif height <= 0.86 or height >= 2.0:
    st.error("Height must be between 0.86 m and 2.0 m.")
else:
    if st.button("Get Recommendations"):
        # Calculate BMI and BMR
        bmi = calculate_bmi(weight, height)
        if bmi is None:
            st.error("Height must be greater than zero.")
        else:
            bmr = calculate_bmr(age, weight, height, gender)
            st.write(f"*Calculated BMI:* {bmi:.2f}")
            st.write(f"*Calculated BMR:* {bmr:.2f} kcal")

            # Provide BMI feedback and suggestions
            if bmi < 18.5:
                target_weight_min = 18.5 * height**2
                st.warning(
                    f"Your BMI is below the normal range (underweight). We suggest gaining weight to reach at least {target_weight_min:.1f} kg (BMI 18.5).")
                if weight_goal == "Lose Weight":
                    st.warning(
                        "Caution: Losing weight is not recommended with your current BMI.")
            elif 18.5 <= bmi < 25:
                st.write("Your BMI is within the normal range.")
            else:  # bmi >= 25
                target_weight_max = 24.9 * height**2
                if 25 <= bmi < 30:
                    st.write(
                        f"Your BMI indicates you are overweight. We suggest losing weight to reach at most {target_weight_max:.1f} kg (BMI 24.9).")
                else:
                    st.write(
                        f"Your BMI indicates obesity. We suggest losing weight to reach at most {target_weight_max:.1f} kg (BMI 24.9).")
                if weight_goal == "Gain Weight":
                    st.warning(
                        "Caution: Gaining weight is not recommended with your current BMI.")

            # Encode gender
            try:
                gender_encoded = gender_label_encoder.transform([gender])[0]
            except ValueError:
                st.error("Error encoding gender.")
                st.stop()

            # Prepare inputs for calories prediction
            inputs = [age, weight, height,
                      gender_encoded, bmi, bmr, activity_level]

            # Predict TDEE
            tdee = predict_calories(inputs)
            st.header("Calorie Requirements")
            st.write(
                f"*Total Daily Energy Expenditure (TDEE):* {tdee:.2f} kcal")

            # Adjust calories based on weight goal
            recommended_calories = adjust_calories(tdee, weight_goal)
            st.write(
                f"*Recommended Calories ({weight_goal}):* {recommended_calories:.2f} kcal")

            # Divide calories into meals
            breakfast_cal, lunch_cal, dinner_cal = divide_calories(
                recommended_calories)
            st.write(f"*Breakfast:* {breakfast_cal:.2f} kcal")
            st.write(f"*Lunch:* {lunch_cal:.2f} kcal")
            st.write(f"*Dinner:* {dinner_cal:.2f} kcal")

            # Predict nutrients and recommend recipes
            try:
                breakfast_nutrients = predict_nutrients(
                    breakfast_cal, meal_models['breakfast'], meal_scalers_X['breakfast'], meal_scalers_y['breakfast'])
                lunch_nutrients = predict_nutrients(
                    lunch_cal, meal_models['lunch'], meal_scalers_X['lunch'], meal_scalers_y['lunch'])
                dinner_nutrients = predict_nutrients(
                    dinner_cal, meal_models['dinner'], meal_scalers_X['dinner'], meal_scalers_y['dinner'])

                breakfast_nutrients_dict = {meal_targets['breakfast'][i]: breakfast_nutrients[i] for i in range(
                    len(meal_targets['breakfast']))}
                lunch_nutrients_dict = {meal_targets['lunch'][i]: lunch_nutrients[i] for i in range(
                    len(meal_targets['lunch']))}
                dinner_nutrients_dict = {meal_targets['dinner'][i]: dinner_nutrients[i] for i in range(
                    len(meal_targets['dinner']))}

                st.header("Meal Recommendations")
                breakfast_recipes = select_meal_recipes(
                    meal_data['breakfast'], breakfast_cal, breakfast_nutrients_dict)
                lunch_recipes = select_meal_recipes(
                    meal_data['lunch'], lunch_cal, lunch_nutrients_dict)
                dinner_recipes = select_meal_recipes(
                    meal_data['dinner'], dinner_cal, dinner_nutrients_dict)

                display_recipes(breakfast_recipes, "Breakfast",
                                breakfast_cal, breakfast_nutrients_dict)
                display_recipes(lunch_recipes, "Lunch",
                                lunch_cal, lunch_nutrients_dict)
                display_recipes(dinner_recipes, "Dinner",
                                dinner_cal, dinner_nutrients_dict)
            except Exception as e:
                st.error(f"Error during prediction or recommendation: {e}")
