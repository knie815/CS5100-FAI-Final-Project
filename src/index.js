import React, { useState, useEffect, memo, useMemo } from "react";
import ReactDOM from "react-dom/client";
import dayjs from "dayjs";
import "./styles.css"; 
import classNames from 'classname';

const App = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [scrollBarPosition, setScrollBarPosition] = useState(0);
  const [currentDate, setCurrentDate] = useState(dayjs()); // set the current date

  const handleMouseDown = () => setIsDragging(true);
  const handleMouseUp = () => {
    setIsDragging(false);
    setScrollBarPosition((prev) => (prev > window.innerWidth * 0.3 ? window.innerWidth : 0));
  };

  const handleMouseMove = (event) => {
    if (isDragging) {
      const width = window.innerWidth - event.clientX - 10;
      setScrollBarPosition(width);
    }
  };

  const handleCloseOverlay = () => {
    setScrollBarPosition(0);
  };

  // update date 
  const handleDateChange = (direction) => {
    if (direction === "next") {
      setCurrentDate((prev) => prev.add(1, 'day')); // Next date
    } else {
      setCurrentDate((prev) => prev.subtract(1, 'day')); // Previous date
    }
  };

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

  // mock data
  const [items, setItems] = useState([
    { id: 1, content: "Item 1", countdown: 3600, isPlaying: false }, // countdown in seconds
    { id: 2, content: "Item 2", countdown: 7200, isPlaying: false },
    { id: 3, content: "Item 3", countdown: 5400, isPlaying: false },
    { id: 4, content: "Item 4", countdown: 1800, isPlaying: false },
    { id: 5, content: "Item 5", countdown: 900, isPlaying: false },
    { id: 6, content: "Item 6", countdown: 1200, isPlaying: false },
    { id: 7, content: "Item 7", countdown: 300, isPlaying: false },
    { id: 8, content: "Item 8", countdown: 1500, isPlaying: false },
    { id: 9, content: "Item 9", countdown: 2400, isPlaying: false },
  ]);
  const deleteItem =(id)=>{
    setItems(prevItems => prevItems.filter(item => item.id !== id));
  }
  const togglePlayPause = (id) => {
    setItems(prevItems =>
      prevItems.map(item =>
        item.id === id ? { ...item, isPlaying: !item.isPlaying } : item
      )
    );
  };
  const shouldShow = useMemo(() => {
    return scrollBarPosition === window.innerWidth
  }, [scrollBarPosition])

  return (
    <>
      <div className="black-overlay" style={{ width: `${scrollBarPosition}px` }}>
        {shouldShow && items.length > 0 ?  (
          <div className='item-list'>
            {items.map(item => (
              <div key={item.id} className="item">
                <div className="date">{currentDate.format("MM.DD")}</div>
                <div className="countdown">{formatCountdown(item.countdown)}</div>
                <button className="play-button" onClick={() => togglePlayPause(item.id)}>
                  {item.isPlaying ? '⏸️' : '▶️'}
                </button>
                <div className="button-bar">
                  <button className="button">Select</button>
                  <button className={classNames("button", "delete")} onClick={()=>{
                    deleteItem(item.id)
                  }}>Delete</button>
                </div>
              </div>
            ))}
          </div>
        ) : null}
        {shouldShow ? (
          <button className="close-button" onClick={handleCloseOverlay}>✖</button>
        ) : null}
      </div>
      <div className="container">
        <div className="red-box">
          <div className="year">{currentDate.year()}</div>
          <div className="date-navigation">
            <button className="arrow" onClick={() => handleDateChange('prev')}>&lt;</button>
            <div className="maindate">{currentDate.format("MM.DD")}</div>
            <button className="arrow" onClick={() => handleDateChange('next')}>&gt;</button>
          </div>
          <textarea className="textarea" placeholder="输入内容..."></textarea>
          <button className="create-button">Create</button>
        </div>
      </div>
      <div className="scroll-bar" style={{ transform: `translateX(${-scrollBarPosition}px)` }} onMouseDown={handleMouseDown}></div>
    </>
  );
};

const formatCountdown = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
