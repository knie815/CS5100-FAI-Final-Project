Running the VibeMatch Application
This application consists of a Python backend and a React frontend, which are connected through a Flask API. Follow these steps to set up and run the application:

Clone the Repository
  First, clone the repository to your local machine:
    git clone https://github.com/knie815/CS5100-FAI-Final-Project.git
    cd CS5100-FAI-Final-Project

Set Up the Backend (Python/Flask)
  Install Dependencies and Navigate to the backend directory i.e flask_app and run the app.py file there.
    pip install -r requirements.txt
    python app.py

The backend server will start, usually on http://localhost:5000.

Set Up the Frontend (React)
  Install Dependencies and Navigate to the frontend directory i.e emotions-ui and install the required dependencies using npm or yarn.
    npm install
    npm start

This will start the React frontend on http://localhost:3000.

The frontend (React) and backend (Flask) communicate via HTTP requests. The Flask server exposes an API at http://localhost:5000, and the React frontend sends requests to this API for emotion detection and music recommendations.

Open http://localhost:3000 in your web browser to interact with the VibeMatch application.
Enter a text prompt in the UI, and the backend will process the text, classify the emotion, and return music recommendations based on the detected emotion.
  

