// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './styles.css';

function App() {
  const [userInput, setUserInput] = useState('');
  const [detectedEmotion, setDetectedEmotion] = useState('');
  const [musicPlaying, setMusicPlaying] = useState(false);
  const [audio, setAudio] = useState(null);

  useEffect(() => {
    if (userInput.trim() !== '') {
      const timeoutId = setTimeout(() => {
        axios
          .post('http://localhost:5000/predict', { text: userInput })
          .then((response) => {
            setDetectedEmotion(response.data.emotion);
          })
          .catch((error) => {
            console.error('Error:', error);
          });
      }, 500);

      return () => clearTimeout(timeoutId);
    } else {
      setDetectedEmotion('');
    }
  }, [userInput]);

  const handlePlayMusic = () => {
    if (audio) {
      audio.pause();
      setAudio(null);
      setMusicPlaying(false);
    } else {
      const emotionMusicMap = {
        sadness: '/music/sad.mp3',
        joy: '/music/happy.mp3',
        love: '/music/love.mp3',
        anger: '/music/angry.mp3',
        fear: '/music/fear.mp3',
        surprise: '/music/surprise.mp3',
      };

      const musicFile = emotionMusicMap[detectedEmotion];

      if (musicFile) {
        const newAudio = new Audio(musicFile);
        newAudio.play();
        setAudio(newAudio);
        setMusicPlaying(true);

        newAudio.onended = () => {
          setMusicPlaying(false);
          setAudio(null);
        };
      } else {
        alert('No music available for this emotion.');
      }
    }
  };

  return (
    <div className="container">
      <h1>VibeMatch</h1>
      <textarea
        placeholder="Type your narrative here..."
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        className="textarea"
      />
      {detectedEmotion && (
        <div className="Vibe">
          Emotion Detected: <strong>{detectedEmotion.toUpperCase()}</strong>
        </div>
      )}
      {detectedEmotion && (
        <button onClick={handlePlayMusic} className="play-button">
          {musicPlaying ? 'Pause Music' : 'Play Music'}
        </button>
      )}
    </div>
  );
}

export default App;
