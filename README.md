AI Teacher Platform
AI Teacher Platform is an interactive, AI-powered educational web application built with Streamlit. It provides personalized learning, analytics, and content creation tools for both students and teachers, leveraging machine learning models trained on real educational datasets.﻿

Features
For Students
Dashboard: View learning summary, quiz scores, completed lessons, and personalized recommendations.
Take Quiz: Attempt quizzes generated from real datasets, with instant feedback and AI-based pass/fail prediction.
Ask Questions: Submit questions and receive AI-generated answers, with automatic difficulty classification.
Learning Resources: Browse and filter resources by subject, type, and difficulty.
Progress Tracking: Visualize progress across subjects, topics, and assessments, with personalized next steps.


For Teachers
Dashboard: Overview of class performance, engagement, and active courses.
Student Analytics: Analyze student performance, learning styles, and behavior clusters.
Content Creation: Create quizzes, assignments, and resources with customizable settings and rubrics.
Review Questions: Analyze question difficulty and distribution.


Project Structure
AI_Teacher_Project/
│
├── datasets/
│   ├── student_quiz_scores.csv
│   ├── student_questions_dataset.csv
│   ├── learning_time_dataset.csv
│   ├── Student Feedback Dataset.csv
│   ├── Student Behavior Dataset.csv
│   └── Topic Recommendation Dataset.csv
│
├── models/
│   ├── quiz_score_model.pkl
│   ├── question_classifier_model.pkl
│   ├── learning_time_model.pkl
│   ├── feedback_model.pkl
│   └── behavior_clustering_model.pkl
│
├── scripts/
│   └── (training scripts)
│
├── app.py
├── requirements.txt
└── README.md


Machine Learning Models
quiz_score_model.pkl: Predicts quiz pass/fail.
question_classifier_model.pkl: Classifies question difficulty.
learning_time_model.pkl: Predicts estimated learning time for topics.
feedback_model.pkl: Predicts feedback usefulness.
behavior_clustering_model.pkl: Clusters students by learning behavior.
All models are trained on the provided datasets and loaded directly in the app using joblib.

