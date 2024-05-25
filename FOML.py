import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Extract relevant features
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['DayOfMonth'] = data['Date'].dt.day
    
    # Encode categorical features (if any)
    data = pd.get_dummies(data, columns=['Category'])
    
    return data

# Function to train and evaluate the model
def train_and_evaluate_model(data):
    # Split data into features (X) and target variable (y)
    X = data.drop(['Date', 'Expense'], axis=1)  # Features
    y = data['Expense']       # Target variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    
    return model

# Function to predict expense for a given day
def predict_expense(model, day_data):
    # Get feature names from the dataset columns
    feature_names = ['DayOfWeek', 'Month', 'DayOfMonth', 'Category_Entertainment', 'Category_Groceries', 'Category_Transportation']
    
    # Reorder columns of day_data to match feature names
    day_data_reordered = day_data[feature_names]
    
    # Predict expense using the trained model
    expense_prediction = model.predict(day_data_reordered)
    return expense_prediction[0]

# Function to suggest budget allocation based on predicted expense
def suggest_budget_allocation(predicted_expense, weekly_budget):
    # Calculate daily budget allocation based on weekly budget
    daily_budget = weekly_budget / 7
    
    # Budget allocation suggestions for each day
    if predicted_expense < daily_budget:
        suggestion = "You can allocate a small budget for this day."
    elif predicted_expense < 2 * daily_budget:
        suggestion = "You may want to allocate a moderate budget for this day."
    else:
        suggestion = "Consider allocating a larger budget for this day."
    
    return suggestion


if __name__ == "__main__":
    # Load and preprocess data
    filename = 'expense_data.csv'
    data = load_and_preprocess_data(filename)
    
    # Train and evaluate the model
    model = train_and_evaluate_model(data)
    
    # Ask user for the specific day
    day_of_week = int(input("Enter the day of the week (0-indexed): "))
    month = int(input("Enter the month (1-indexed): "))
    day_of_month = int(input("Enter the day of the month: "))
    
    # Prepare data for prediction
    day_data = pd.DataFrame({
        'DayOfWeek': [day_of_week],
        'Month': [month],
        'DayOfMonth': [day_of_month],
        'Category_Entertainment': [0],
        'Category_Groceries': [1],
        'Category_Transportation': [0]
    })
    
    # Ask user for the budget for the whole week
    while True:
        try:
            weekly_budget = float(input("Enter your budget for the whole week: $"))
            break
        except ValueError:
            print("Please enter a valid number.")

    # Predict expense for the provided day
    expense_prediction = predict_expense(model, day_data)
    print(f'Predicted expense for the provided day: ${expense_prediction:.2f}')
    
    # Suggest budget allocation based on predicted expense and weekly budget
    budget_allocation_suggestion = suggest_budget_allocation(expense_prediction, weekly_budget)
    print(f'Suggestion: {budget_allocation_suggestion}')



