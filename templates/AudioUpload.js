import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import 'wavesurfer.js/dist/wavesurfer.css';

function AudioUpload() {
  const [selectedMusic, setSelectedMusic] = useState('');
  const fileInputRef = useRef();
  const wavesurferRef = useRef(null);

  const handleMusicChange = () => {
    const selectedValue = fileInputRef.current.value;
    setSelectedMusic(selectedValue);
  };

  const handleFileUpload = async () => {
    const formData = new FormData();
    formData.append('newMusic', fileInputRef.current.files[0]);
    formData.append('newFileName', selectedMusic);

    try {
      const response = await axios.post('/upload_music', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log(response.data);
    } catch (error) {
      console.error('Error uploading music:', error);
    }
  };

  useEffect(() => {
    wavesurferRef.current = WaveSurfer.create({
      container: '#waveform',
      waveColor: 'violet',
      progressColor: 'purple',
      cursorColor: 'navy',
    });

    wavesurferRef.current.load('/static/audio/cardigan.mp3');
  }, []);

  return (
    <div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleMusicChange}
        accept=".mp3, .wav, .ogg"
        style={{ display: 'none' }}
      />
      <button onClick={() => fileInputRef.current.click()}>Select Music</button>
      <button onClick={handleFileUpload}>Upload Music</button>
      <select onChange={handleMusicChange} value={selectedMusic}>
        <option value="clairdelune.wav">Clair de Lune</option>
        <option value="cardigan.mp3">Cardigan-Taylor Swift</option>
        <option value="ny_short.m4a">Empire State of Mind</option>
        <option value="seleners.wav">Beauty and a Beat-JB</option>
        <option value="baby_shark.m4a">Baby Shark</option>
      </select>
      <div id="waveform"></div>
    </div>
  );
}

export default AudioUpload;
