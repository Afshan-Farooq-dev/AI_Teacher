import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
data = pd.read_csv('datasets/Student Behavior Dataset.csv')

# Convert Focus_Level to numeric
data['Focus_Level'] = data['Focus_Level'].map({'Low':1, 'Medium':2, 'High':3})

# Features for clustering
X = data[['Login_Frequency', 'Avg_Session_Time', 'Focus_Level', 'Content_Viewed']]

# Train KMeans with 3 clusters
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, 'models/behavior_clustering_model.pkl')

print('Behavior clustering model trained and saved.')
