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
          'Content-Type': 'multipart/form-data'
        }
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
      cursorColor: 'navy'
    });
    wavesurferRef.current.load('/static/audio/cardigan.mp3');
  }, []);
  return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("input", {
    type: "file",
    ref: fileInputRef,
    onChange: handleMusicChange,
    accept: ".mp3, .wav, .ogg",
    style: {
      display: 'none'
    }
  }), /*#__PURE__*/React.createElement("button", {
    onClick: () => fileInputRef.current.click()
  }, "Select Music"), /*#__PURE__*/React.createElement("button", {
    onClick: handleFileUpload
  }, "Upload Music"), /*#__PURE__*/React.createElement("select", {
    onChange: handleMusicChange,
    value: selectedMusic
  }, /*#__PURE__*/React.createElement("option", {
    value: "clairdelune.wav"
  }, "Clair de Lune"), /*#__PURE__*/React.createElement("option", {
    value: "cardigan.mp3"
  }, "Cardigan-Taylor Swift"), /*#__PURE__*/React.createElement("option", {
    value: "ny_short.m4a"
  }, "Empire State of Mind"), /*#__PURE__*/React.createElement("option", {
    value: "seleners.wav"
  }, "Beauty and a Beat-JB"), /*#__PURE__*/React.createElement("option", {
    value: "baby_shark.m4a"
  }, "Baby Shark")), /*#__PURE__*/React.createElement("div", {
    id: "waveform"
  }));
}
export default AudioUpload;
