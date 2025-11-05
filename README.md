# Cognitive Load Assessment System

## ML-Powered Text Readability and Difficulty Classification

This repository contains a machine learning project designed to analyze and classify the cognitive load or difficulty level of a given text. The system classifies content into six distinct educational levels, making it a valuable tool for educators, content creators, and curriculum designers who need to match text complexity with a target audience.

The project uses an ensemble of machine learning models trained on linguistic features extracted from the text. It is deployed as a simple web application with a Flask backend API and an interactive HTML/JavaScript frontend.

## Features

* **Text Difficulty Classification:** Automatically classifies input text into one of six levels:
    * Grade till 6th
    * Grade 6-8
    * Grade 9-10
    * Grade 11-12
    * Undergraduate
    * Postgraduate
* **Linguistic Feature Extraction:** Leverages `NLTK` and `textstat` libraries to extract a wide range of linguistic and readability features (e.g., Flesch-Kincaid grade, sentence complexity, word length, lexical diversity).
* **Ensemble ML Model:** Utilizes a highly accurate ensemble model (combining Random Forest and Gradient Boosting) for robust prediction.
* **Web Dashboard:** An interactive and user-friendly web interface (`Index.html`) for pasting text and receiving an instant analysis of its cognitive load.
* **REST API:** A lightweight Flask backend (`App.py`) that serves the trained model and provides a simple API endpoint for analysis.

## Technologies Used

* **Backend:** Python, Flask, Flask-CORS
* **Frontend:** HTML, CSS, JavaScript
* **Machine Learning:** Scikit-learn (RandomForestClassifier, GradientBoostingClassifier)
* **Data Processing:** Pandas, NumPy
* **NLP & Feature Extraction:** NLTK, textstat

## How It Works

1.  **Input:** A user pastes a block of text into the web dashboard.
2.  **API Request:** The frontend sends the text to the `/api/analyze` endpoint of the Flask server.
3.  **Feature Extraction:** The backend `CognitiveLoadClassifier` uses `NLTK` and `textstat` to process the text and extract a numerical feature vector representing its linguistic complexity.
4.  **Prediction:** This feature vector is fed into the pre-trained ensemble model (`cognitiveload_model.pkl`).
5.  **Output:** The model predicts the most likely difficulty level (e.g., "Grade 9-10") and a confidence score, which are returned to the frontend and displayed to the user.

## Dataset

The model was trained on the `Book_Dataset.csv` file, which contains various text samples, each labeled with one of the six corresponding difficulty levels.

![Output Image 1](https://github.com/devray11/Automated-Readability-Assessment-for-Educational-Content-using-Deep-Learning/blob/d6caf8d6d3f6bdc8d269762a666c3c7f1aa495ae/Output-Image-1.jpg)

![Output Image 2](https://github.com/devray11/Automated-Readability-Assessment-for-Educational-Content-using-Deep-Learning/blob/d6caf8d6d3f6bdc8d269762a666c3c7f1aa495ae/Output-Image-2.png)

![Output Image 3](https://github.com/devray11/Automated-Readability-Assessment-for-Educational-Content-using-Deep-Learning/blob/d6caf8d6d3f6bdc8d269762a666c3c7f1aa495ae/Output-Image-3.png)

![Output Image 4](https://github.com/devray11/Automated-Readability-Assessment-for-Educational-Content-using-Deep-Learning/blob/d6caf8d6d3f6bdc8d269762a666c3c7f1aa495ae/Output-Image-4.png)

