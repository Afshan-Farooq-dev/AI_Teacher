import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('datasets/student_quiz_scores.csv')
print(data.columns)

# Prepare features and target
# Convert categorical columns to numeric using pd.get_dummies
X = pd.get_dummies(data[['Topic_Name', 'Difficulty_Level', 'Attempt_Number', 'Time_Taken(mins)']])
y = data['Passed(Yes/No)'].map({'Yes':1, 'No':0})  # Convert target to 0/1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'models/quiz_score_model.pkl')
