# VibeMatch - Real-Time Emotion Detection

## Overview
VibeMatch is a web application that detects emotions from user-generated text and recommends music that matches the detected emotional tone. By leveraging Natural Language Processing (NLP) techniques and machine learning models, the system is designed to bridge the gap between textual emotional expression and musical experiences. The application uses a **Word2Vec + Linear Discriminant Analysis (LDA)** pipeline to classify emotions, providing both high performance and computational efficiency.

## Features
- **Real-time Emotion Detection**: Detects emotions from user input (e.g., stories or narratives).
- **Emotion-based Music Recommendations**: Recommends music that aligns with the detected emotions.
- **Fast and Responsive**: Music recommendations are returned with an average latency of 1.2 seconds.
- **Web-based Application**: Built with a Python backend (Flask) and a React frontend.

## Technologies
- **Backend**: Python, Flask
- **Frontend**: React
- **NLP**: Word2Vec, Linear Discriminant Analysis (LDA)
- **Machine Learning**: BERT (for evaluation), CNN, Decision Tree, Naive Bayes
- **Data**: Emotion-labeled text dataset (416,809 samples) with six emotion categories: Joy, Sadness, Anger, Fear, Love, and Surprise.

## Demo
You can watch a video demo of the VibeMatch application here:  
- [YouTube Demo 1](https://www.youtube.com/watch?v=kVTiYsABm4o&t=1s)
- [YouTube Demo 2](https://www.youtube.com/watch?v=0ESAUNm3Ia8)

## Setup Instructions

Follow these steps to set up and run the VibeMatch application locally.

### Prerequisites
- Python 3.x
- Node.js (for React frontend)
- npm (or yarn)

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/knie815/CS5100-FAI-Final-Project.git
cd CS5100-FAI-Final-Project
```

### 2. Set Up the Backend (Flask + Python)
Navigate to the `flask_app` directory and install the required dependencies:
```bash
cd flask_app
pip install -r requirements.txt
```
Run the Flask backend server:
```bash
python app.py
```
The backend server should now be running on `http://localhost:5000`.

### 3. Set Up the Frontend (React)
Navigate to the `emotions-ui` directory and install the frontend dependencies:
```bash
cd emotions-ui
npm install
```
Start the React frontend:
```bash
npm start
```
This will start the React application on `http://localhost:3000`.

### 4. Interact with the Application
- Open `http://localhost:3000` in your browser.
- Enter a text prompt in the UI, and the backend will process the text, classify the emotion, and return music recommendations based on the detected emotion.

## Application Flow

1. **User Input**: The user types a story or text in the provided input box.
2. **Emotion Detection**: The backend processes the input using a pre-trained NLP model (Word2Vec + LDA) to detect the emotion.
3. **Music Recommendation**: Based on the detected emotion, the backend queries a music dataset and returns the top music tracks that match the emotion.
4. **Music Playback**: Once the music is fetched, the user can interact with the frontend and play the recommended songs.

## Models and Experimentation
In this project, we experimented with several models for emotion classification:
- **Naive Bayes with Bag-of-Words**: Initial experimentation showed an accuracy of 83%.
- **Word2Vec + Decision Tree**: The accuracy was lower than expected, likely due to overfitting.
- **Convolutional Neural Networks (CNN)**: Achieved an accuracy of 91%.
- **Linear Discriminant Analysis (LDA)**: The best trade-off between performance and computational efficiency.
- **BERT**: The most accurate model with an accuracy of 95%, but resource-intensive.

Ultimately, we chose the **Word2Vec + LDA** model, which offered a balance between performance and computational feasibility, making it ideal for real-time applications.

## Results

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Decision Tree      | 47%      | 0.42      | 0.45   | 0.43     |
| CNN                | 91%      | 0.90      | 0.92   | 0.91     |
| LDA                | 74%      | 0.72      | 0.74   | 0.73     |
| BERT               | 95%      | 0.94      | 0.95   | 0.94     |

The **BERT** model provided the best accuracy, but for the sake of efficiency, **Word2Vec + LDA** was selected for the application.

## Future Work
- **Emotion Granularity**: Extend the model to detect a broader range of emotions beyond the current six categories.
- **Cross-lingual Support**: Introduce support for multiple languages to make the system more accessible.
- **Personalized Recommendations**: Adapt music suggestions based not only on detected emotions but also user preferences and listening history.
- **User Feedback**: Implement functionality where users can provide feedback on music recommendations, refining the system's suggestions over time.
- **User Research**: Investigate integrating music therapy for emotional support, leveraging journal writing as a method for emotional processing.

## Limitations
- **Emotion Granularity**: The system currently supports only six emotions. More fine-grained emotion detection could improve the experience.
- **Dataset Imbalance**: Some emotions (like 'surprise') are underrepresented, affecting classification performance.
- **Computational Resources**: Models like BERT require significant resources, which limit their use in real-time scenarios.

## Team Contributions
- **Sarakshi Phate**: Tested preprocessing techniques and variations of LDA and CNN models.
- **Yuxiang Nie**: Worked on the initial Word2Vec model with Decision Tree and helped test the BERT model.
- **Nithin Subramoniam**: Managed data merging, multiple models (LDA, Word2Vec, BERT), backend and real-time emotion detection integration.
- **Purvaja Narayana**: Curated the music dataset, integrated frontend and backend, and tested CNN and Naive Bayes models.
- **Yuxi Zhou**: Designed user flow, built frontend UI, and tested the LDA model.



