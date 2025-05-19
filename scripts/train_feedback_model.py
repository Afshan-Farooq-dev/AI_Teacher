import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('datasets/Student Feedback Dataset.csv')

# Target: Usefulness_Rating > 3 -> Good(1), else Bad(0)
data['Usefulness_Binary'] = (data['Usefulness_Rating'] > 3).astype(int)

# Features
X = pd.get_dummies(data[['Lesson_Name', 'Clarity_Rating', 'Difficulty_Feedback']])
y = data['Usefulness_Binary']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'models/feedback_model.pkl')
