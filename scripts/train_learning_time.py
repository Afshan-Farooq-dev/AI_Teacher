import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
data = pd.read_csv('datasets/learning_time_dataset.csv')

# Prepare features and target
# Convert categorical to numeric
X = pd.get_dummies(data[['Student_ID', 'Lesson_Name', 'Completion_Status', 'Revision_Count']])
y = data['Time_Spent(mins)']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save model
joblib.dump(model, 'models/learning_time_model.pkl')
